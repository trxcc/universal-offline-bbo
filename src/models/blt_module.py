from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers.tokenization_utils_base import BatchEncoding

from src.utils import RankedLogger
from src.utils.io_utils import load_task_names

log = RankedLogger(__name__, rank_zero_only=True)


class BLTEmbedModule(LightningModule):

    def __init__(
        self,
        embedder: nn.Module,
        embedder_output_dim: int,
        regressor: nn.Module,
        regressor_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        data_dir: Path,
        task_names: List[str],
        entropy_model: Any,
        entropy_model_checkpoint: str,
        entropy_threshold: float,
        embedder_optimizer: Optional[torch.optim.Optimizer] = None,
        metadata_embedder: Optional[nn.Module] = None,
        metadata_embedder_output_dim: Optional[int] = None,
        metadata_projector: Optional[nn.Module] = None,
        metadata_projector_output_dim: Optional[int] = None,
        metadata_embedder_optimizer: Optional[torch.optim.Optimizer] = None,
        cat_metadata: Optional[bool] = False,
        from_pretrained: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.automatic_optimization = False
        self.task_names = load_task_names(task_names, data_dir)

        self.embedder = embedder
        self.embedder.init_weights()
        self.embedder_output_dim = embedder_output_dim
        self.regressor = regressor
        self.cat_metadata = cat_metadata
        self.entropy_model = entropy_model
        self.entropy_model.load_from_checkpoint(entropy_model_checkpoint)
        self.entropy_threshold = entropy_threshold
        self.has_metadata = (
            (metadata_embedder is not None)
            and (metadata_embedder_output_dim is not None)
            and (metadata_projector_output_dim is not None)
        )
        self.optimize_metadata_embedder = (
            self.has_metadata and metadata_embedder_optimizer is not None
        )

        if self.has_metadata:
            self.metadata_embedder = metadata_embedder
            self.metadata_embedder_output_dim = metadata_embedder_output_dim
            if metadata_projector is None:
                metadata_projector = nn.Linear(
                    in_features=metadata_embedder_output_dim,
                    out_features=metadata_projector_output_dim,
                )
            self.metadata_projector = metadata_projector
            self.metadata_projector_output_dim = metadata_projector_output_dim

        if not self.has_metadata:
            self.batch_norm = nn.BatchNorm1d(embedder_output_dim)
        else:
            self.batch_norm = nn.BatchNorm1d(
                embedder_output_dim + metadata_projector_output_dim
            )

        self.criterion = nn.MSELoss()

        self.train_rank_corr = {}
        self.val_rank_corr = {}
        self.val_rank_corr_avg_best = {}

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def _emb_metadata(self, m: Tuple[BatchEncoding]) -> torch.Tensor:
        context = (
            torch.no_grad() if not self.optimize_metadata_embedder else nullcontext()
        )
        with context:
            encoded_input = m.to(self.device)
            emb_m = self.metadata_embedder(**encoded_input)
            emb_m = self._mean_pooling(emb_m, encoded_input["attention_mask"])
        return emb_m

    def forward(
        self,
        x: Tuple[BatchEncoding],
        m: Tuple[BatchEncoding],
        patch_ids: Tuple[BatchEncoding],
    ) -> torch.Tensor:
        # assert x.dtype in [torch.long, torch.int64]
        # assert entropy_patch_start_idx.dtype in [torch.long, torch.int64]
        # assert torch.all(entropy_patch_start_idx >= 0)
        x_emb = x.to(self.device)
        patch_ids = patch_ids.to(self.device)
        try:
            x_emb = self.embedder(x_emb, patch_ids=patch_ids)
        except Exception as e:
            print(e)
            print("module forward error")
            print(patch_ids.shape)
            print(x.shape)
            print(self.embedder.tok_embeddings.weight.shape)
            raise e

        if self.has_metadata:
            m_emb = self._emb_metadata(m)
            m_emb = self.metadata_projector(m_emb)
            x_emb = torch.cat((x_emb, m_emb), dim=1)

        x_emb = self.batch_norm(x_emb)
        preds = self.regressor(x_emb)
        return preds

    def _mean_pooling(
        self, model_output: Tuple[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # First element of model_output contains all token embeddings
        token_embeddings: torch.Tensor = model_output[
            0
        ]  # Shape: [batch_size, sequence_length, hidden_size]
        input_mask_expanded: torch.Tensor = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )  # Shape: [batch_size, sequence_length, hidden_size]

        sum_embeddings: torch.Tensor = torch.sum(
            token_embeddings * input_mask_expanded, 1
        )  # Shape: [batch_size, hidden_size]
        sum_mask: torch.Tensor = torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )  # Shape: [batch_size, hidden_size]

        return sum_embeddings / sum_mask  # Shape: [batch_size, hidden_size]

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def model_step(
        self, batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["text"]
        y = batch["value"]
        m = batch["metadata"]
        task_names = batch["task_names"]
        tokens_length = batch["tokens_length"]
        patch_ids = self.get_entropy_patch_start_idx(x, tokens_length)
        preds = self.forward(x, m, patch_ids)
        loss = self.criterion(preds.squeeze(), y.squeeze())
        return loss, preds, y, task_names

    def get_entropy_patch_start_idx(
        self, text_tokens: torch.Tensor, tokens_length: torch.Tensor
    ) -> torch.Tensor:
        text_tokens = text_tokens.to(self.device)
        with torch.no_grad():
            logits = self.entropy_model(text_tokens)
        entropy = self.entropy(logits)
        start_idx = self.get_entropy_patch_idx(
            entropy, self.entropy_threshold, tokens_length
        )
        return start_idx

    def get_entropy_patch_idx(
        self, entropy: torch.Tensor, threshold: float, tokens_length: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len = entropy.shape

        start_idx = torch.zeros_like(entropy, dtype=torch.bool)  # [bsz, seq_len]
        start_idx[:, 0] = True

        diff = entropy[:, 1:] - entropy[:, :-1]
        start_idx[:, 1:] = diff > threshold

        batch_indices = torch.arange(bsz, device=entropy.device)
        start_idx[batch_indices, tokens_length.squeeze(-1) - 1] = True

        start_idx_int = start_idx.long()  # [bsz, seq_len]

        result = start_idx_int.cumsum(dim=1) - 1  # [bsz, seq_len]

        return result

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
        return entropy

    def training_step(
        self,
        batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.optimize_metadata_embedder:
            opt_regress, opt_embed, opt_m_embed = self.optimizers()
            opt_m_embed.zero_grad()
        else:
            opt_regress, opt_embed = self.optimizers()
        opt_regress.zero_grad()
        opt_embed.zero_grad()

        loss, preds, targets, task_names = self.model_step(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=0.5)

        if self.optimize_metadata_embedder:
            torch.nn.utils.clip_grad_norm_(
                self.metadata_embedder.parameters(), max_norm=0.5
            )

        opt_embed.step()
        opt_regress.step()
        if self.optimize_metadata_embedder:
            opt_m_embed.step()

        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train_loss",
        )

        return loss

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % 5 == 0:
            self.compute_rank_corr(self.trainer.train_dataloader, "train")

    def validation_step(
        self,
        batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]],
        batch_idx: int,
    ) -> None:
        loss, preds, targets, task_names = self.model_step(batch)
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="val_loss",
        )

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % 5 == 0:
            self.compute_rank_corr(self.trainer.val_dataloaders, "val")

    def test_step(self) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if stage == "fit":
            for task_name in self.task_names:
                if task_name not in self.train_rank_corr:
                    self.train_rank_corr[task_name] = SpearmanCorrCoef()
                if task_name not in self.val_rank_corr:
                    self.val_rank_corr[task_name] = SpearmanCorrCoef()
                if task_name not in self.val_rank_corr_avg_best:
                    self.val_rank_corr_avg_best[task_name] = MaxMetric()
        if self.hparams.compile and stage == "fit":
            self.regressor = torch.compile(self.regressor)
            self.embedder = torch.compile(self.embedder)

            if self.has_metadata:
                self.metadata_embedder = torch.compile(self.metadata_embedder)
                self.metadata_projector = torch.compile(self.metadata_projector)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.has_metadata:
            regressor_params = chain(
                self.trainer.model.parameters(),
                self.metadata_projector.parameters(),
            )
            if self.optimize_metadata_embedder:
                metadata_embedder_optimizer = self.hparams.metadata_embedder_optimizer(
                    params=self.metadata_embedder.parameters()
                )
        else:
            regressor_params = self.trainer.model.parameters()

        optimizer = self.hparams.regressor_optimizer(params=regressor_params)
        embedder_optimizer = self.hparams.embedder_optimizer(
            params=self.embedder.parameters()
        )

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            optimizers = [
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                },
                {
                    "optimizer": embedder_optimizer,
                },
            ]
        else:
            optimizers = [optimizer, embedder_optimizer]

        if self.optimize_metadata_embedder:
            optimizers.append({"optimizer": metadata_embedder_optimizer})

        return optimizers

    def compute_rank_corr(self, dataloader, prefix="train"):

        self.eval()
        task_preds = {task: [] for task in self.task_names}
        task_targets = {task: [] for task in self.task_names}
        if prefix == "train":
            compute_rank_corr = self.train_rank_corr
        else:
            compute_rank_corr = self.val_rank_corr
            compute_rank_corr_avg_best = self.val_rank_corr_avg_best
        with torch.no_grad():
            for batch in dataloader:
                x = batch["text"]
                y = batch["value"]
                m = batch["metadata"]
                task_names = batch["task_names"]
                tokens_length = batch["tokens_length"]
                patch_ids = self.get_entropy_patch_start_idx(x, tokens_length)

                y = y.to(self.device)
                preds = self.forward(x, m, patch_ids)
                for i, task_name in enumerate(task_names):
                    task_preds[task_name].append(preds[i].squeeze())
                    task_targets[task_name].append(y[i].squeeze())

        all_corrs = []
        for task_name in self.task_names:
            if task_preds[task_name]:
                task_pred = torch.stack(task_preds[task_name])
                task_target = torch.stack(task_targets[task_name])
                rank_corr = compute_rank_corr[task_name](task_pred, task_target)
                all_corrs.append(rank_corr)
                self.log(
                    f"{prefix}/rank_corr_{task_name}",
                    rank_corr,
                    sync_dist=True,
                    prog_bar=False,
                    metric_attribute=f"{prefix}/rank_corr_{task_name}",
                )

        if all_corrs:
            avg_corr = torch.stack(all_corrs).mean()
            self.log(
                f"{prefix}/rank_corr_avg",
                avg_corr,
                sync_dist=True,
                prog_bar=True,
                metric_attribute=f"{prefix}/rank_corr_avg",
            )
            if prefix == "val":
                compute_rank_corr_avg_best[task_name](avg_corr)
                self.log(
                    f"{prefix}/rank_corr_avg_best/{task_name}",
                    compute_rank_corr_avg_best[task_name].compute(),
                    sync_dist=True,
                    prog_bar=False,
                    metric_attribute=f"{prefix}/rank_corr_avg_best/{task_name}",
                )

        self.train()
