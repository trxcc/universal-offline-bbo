from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers.tokenization_utils_base import BatchEncoding

from src.utils import RankedLogger
from src.utils.io_utils import load_task_names

log = RankedLogger(__name__, rank_zero_only=True)


class EmbedRegressorLatentModule(LightningModule):

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
        non_shuffled_datamodule: LightningDataModule,
        embedder_optimizer: Optional[torch.optim.Optimizer] = None,
        metadata_embedder: Optional[nn.Module] = None,
        metadata_embedder_output_dim: Optional[int] = None,
        metadata_embedder_optimizer: Optional[torch.optim.Optimizer] = None,
        cat_metadata: bool = False,
        from_pretrained: bool = True,
        finetune_embedder: bool = True,
        finetune_interval: int = 10,
        num_finetune_epochs: int = 3,
        temperature: float = 0.07,
        if_use_detach_normalization: bool = False
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        if not from_pretrained:

            def init_weights(m):
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.zeros_(m.bias)

            embedder.apply(init_weights)

        self.automatic_optimization = False
        self.task_names = load_task_names(task_names, data_dir)

        self.embedder = embedder
        self.embedder_output_dim = embedder_output_dim
        self.regressor = regressor
        self.cat_metadata = cat_metadata
        self.finetune_embedder = finetune_embedder
        self.has_metadata = (metadata_embedder is not None) and (
            metadata_embedder_output_dim is not None
        )
        self.optimize_metadata_embedder = False

        if self.has_metadata:
            self.metadata_embedder = metadata_embedder
            self.metadata_embedder_output_dim = metadata_embedder_output_dim

        self.batch_norm = nn.BatchNorm1d(embedder_output_dim)

        self.criterion = nn.MSELoss()

        self.train_rank_corr = {}
        self.val_rank_corr = {}
        self.val_rank_corr_avg_best = {}

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_mse = MeanMetric()
        self.train_lip_loss = MeanMetric()
        self.train_con_loss = MeanMetric()

        self.finetune_interval = finetune_interval
        self.num_finetune_epochs = num_finetune_epochs
        self.temperature = temperature
        self.in_finetune_mode = False
        self.current_finetune_epoch = 0
        self.non_shuffled_datamodule = non_shuffled_datamodule

        self.projection_head = nn.Sequential(
            nn.Linear(embedder_output_dim, embedder_output_dim),
            nn.ReLU(),
            nn.Linear(embedder_output_dim, 128),
        )

        self.metadata_projection_head = nn.Sequential(
            nn.Linear(metadata_embedder_output_dim, metadata_embedder_output_dim),
            nn.ReLU(),
            nn.Linear(metadata_embedder_output_dim, 128),
        )

        if not from_pretrained:
            self.projection_head.apply(init_weights)
            self.metadata_projection_head.apply(init_weights)

    def contrastive_loss(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        metadata_sim = torch.matmul(embeddings2, embeddings2.T)
        metadata_sim_max, _ = metadata_sim.max(dim=1, keepdim=True)
        metadata_sim_min, _ = metadata_sim.min(dim=1, keepdim=True)
        metadata_sim = (metadata_sim - metadata_sim_min) / (
            metadata_sim_max - metadata_sim_min + 1e-10
        )

        similarity_matrix = torch.matmul(embeddings1, embeddings1.T)
        similarity_matrix = similarity_matrix / self.temperature

        log_prob = F.log_softmax(similarity_matrix, dim=1)

        mask = ~torch.eye(
            embeddings1.shape[0], dtype=torch.bool, device=embeddings1.device
        )
        loss = -(metadata_sim[mask] * log_prob[mask]).mean()

        return loss

    def lipschitz_loss(self, z, y, recon_weight=None):
        z = z.to(self.device)
        y = y.to(self.device)
        
        # if all ys are the same, return the mean of all zs
        if torch.all(y == y[0]):
            dif_z = torch.sqrt(
                torch.sum((z.unsqueeze(1) - z.unsqueeze(0)) ** 2, dim=2) + 1e-10
            )
            return dif_z.mean()
        
        dif_y = (y.unsqueeze(1) - y.unsqueeze(0)).squeeze(-1)
        dif_z = torch.sqrt(
            torch.sum((z.unsqueeze(1) - z.unsqueeze(0)) ** 2, dim=2) + 1e-10
        )
        
        lips = abs(dif_y / (dif_z + 1e-10))
        ratio = lips - torch.median(lips)
        ratio = ratio[ratio > 0]
        
        if len(ratio) == 0:
            return torch.tensor(0.0, device=self.device)
            
        loss = ratio.mean()
        return loss

    def finetune_embedder_step(
        self,
        batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]],
        non_shuffled_batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]],
    ) -> torch.Tensor:

        x = batch["text"]
        m = batch["metadata"]

        encoded_input = x.to(self.device)
        x_embeddings = self.embedder(**encoded_input)
        x_embeddings = self._mean_pooling(x_embeddings, encoded_input["attention_mask"])
        x_projected = self.projection_head(x_embeddings)

        with torch.no_grad():
            m_embeddings = self._emb_metadata(m)
        m_projected = self.metadata_projection_head(m_embeddings)

        loss = self.contrastive_loss(x_projected, m_projected)

        x_n = non_shuffled_batch["text"]
        y_n = non_shuffled_batch["value"]
        task_name_n = non_shuffled_batch["task_names"]

        encoded_input_n = x_n.to(self.device)
        x_embeddings_n = self.embedder(**encoded_input_n)
        x_embeddings_n = self._mean_pooling(
            x_embeddings_n, encoded_input_n["attention_mask"]
        )

        # Combine
        task_data = {}
        for i in range(len(task_name_n)):
            task = task_name_n[i]
            if task not in task_data:
                task_data[task] = {"text": [], "value": []}

            task_data[task]["text"].append(x_embeddings_n[i])
            if y_n.dim() > 1:
                task_data[task]["value"].append(y_n[i : i + 1])
            else:
                task_data[task]["value"].append(y_n[i])

        loss_lip = 0
        for task in task_data:
            if len(task_data[task]["text"]) > 0:
                task_data[task]["text"] = torch.stack(task_data[task]["text"], dim=0)

            if len(task_data[task]["value"]) > 0:
                try:
                    task_data[task]["value"] = torch.cat(
                        task_data[task]["value"], dim=0
                    )
                except:
                    task_data[task]["value"] = torch.tensor(task_data[task]["value"])

            loss_lip += self.lipschitz_loss(
                z=task_data[task]["text"], y=task_data[task]["value"]
            ) * (len(task_data[task]["value"]) / len(y_n))

        self.train_lip_loss(loss_lip)
        self.log(
            "train/lip_loss",
            self.train_lip_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train_lip_loss",
        )

        self.train_con_loss(loss)
        self.log(
            "train/con_loss",
            self.train_con_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train_con_loss",
        )

        if self.hparams.if_use_detach_normalization:
            assert 0
            return loss, loss_lip
        else:
            return loss * 0.1

    def _emb_metadata(self, m: Tuple[BatchEncoding]) -> torch.Tensor:
        context = (
            torch.no_grad() if not self.optimize_metadata_embedder else nullcontext()
        )
        with context:
            encoded_input = m.to(self.device)
            emb_m = self.metadata_embedder(**encoded_input)
            emb_m = self._mean_pooling(emb_m, encoded_input["attention_mask"])
        return emb_m

    def forward(self, x: Tuple[BatchEncoding], m: Tuple[BatchEncoding]) -> torch.Tensor:
        context = torch.no_grad() if not self.finetune_embedder else nullcontext()
        with context:
            encoded_input = x.to(self.device)
            x_emb = self.embedder(**encoded_input)
            x_emb = self._mean_pooling(x_emb, encoded_input["attention_mask"])

        x_emb = self.batch_norm(x_emb)
        preds = self.regressor(x_emb)
        return preds

    def _mean_pooling(
        self, model_output: Tuple[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        context = torch.no_grad() if not self.finetune_embedder else nullcontext()
        with context:
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

        preds = self.forward(x, m)
        loss = self.criterion(preds.squeeze(), y.squeeze())
        return loss, preds, y, task_names

    def training_step(
        self,
        batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]],
        batch_idx: int,
    ) -> torch.Tensor:
        try:
            non_shuffled_batch = next(self.non_shuffled_train_iter)
        except:
            self.non_shuffled_train_iter = iter(
                self.non_shuffled_datamodule.train_dataloader()
            )
            non_shuffled_batch = next(self.non_shuffled_train_iter)

        latent_space_loss = self.finetune_embedder_step(batch, non_shuffled_batch)

        if self.optimize_metadata_embedder:
            opt_regress, opt_embed, opt_m_embed = self.optimizers()
            opt_m_embed.zero_grad()
        elif self.finetune_embedder:
            opt_regress, opt_embed = self.optimizers()
            opt_embed.zero_grad()
        else:
            opt_regress = self.optimizers()
        opt_regress.zero_grad()

        loss, preds, targets, task_names = self.model_step(batch)

        self.train_mse(loss)
        self.log(
            "train/mse",
            self.train_mse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train_mse",
        )
        if self.hparams.if_use_detach_normalization:
            loss1 = loss 
            loss2, loss3 = latent_space_loss
            if torch.isnan(loss1) or torch.isnan(loss2) or torch.isnan(loss3):
                print(loss1, loss2, loss3)
                print(preds, targets)
                assert 0
            # loss = loss1 + loss2 / (loss2 / (loss1.detach() + 1e-10) + 1e-10).detach() + loss3 / (loss3 / (loss1.detach() + 1e-10) + 1e-10).detach()
            loss = loss1 / (loss1.detach() + 1e-10) + loss2 / (loss2.detach() + 1e-10) + loss3 / (loss3.detach() + 1e-10)
        else:
            loss = loss + latent_space_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=0.5)
        if self.finetune_embedder:
            torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), max_norm=0.5)
        if self.optimize_metadata_embedder:
            torch.nn.utils.clip_grad_norm_(
                self.metadata_embedder.parameters(), max_norm=0.5
            )

        opt_regress.step()
        if self.finetune_embedder:
            opt_embed.step()
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
            self.non_shuffled_datamodule.setup("fit")
            loader = self.non_shuffled_datamodule.train_dataloader()
            self.non_shuffled_train_iter = iter(loader)
            log.info("Finish iterate non_shuffled dataloader")

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
            self.projection_head = torch.compile(self.projection_head)
            self.metadata_projection_head = torch.compile(self.metadata_projection_head)

            if self.has_metadata:
                self.metadata_embedder = torch.compile(self.metadata_embedder)
                # self.metadata_projector = torch.compile(self.metadata_projector)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.has_metadata:
            regressor_params = self.trainer.model.parameters()
            if self.optimize_metadata_embedder:
                metadata_embedder_optimizer = self.hparams.metadata_embedder_optimizer(
                    params=self.metadata_embedder.parameters()
                )
        else:
            regressor_params = self.trainer.model.parameters()

        optimizer = self.hparams.regressor_optimizer(params=regressor_params)

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
            ]
        else:
            optimizers = [{"optimizer": optimizer}]

        if self.hparams.finetune_embedder:
            embedder_optimizer = self.hparams.embedder_optimizer(
                params=chain(
                    self.embedder.parameters(),
                    self.projection_head.parameters(),
                    self.metadata_projection_head.parameters(),
                )
            )
            optimizers.append({"optimizer": embedder_optimizer})

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

                y = y.to(self.device)
                preds = self.forward(x, m)
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
