from contextlib import nullcontext
from itertools import chain
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers.tokenization_utils_base import BatchEncoding

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class EmbedRegressorModule(LightningModule):

    def __init__(
        self,
        tokenizer: Any,
        embedder: nn.Module,
        embedder_output_dim: int,
        regressor: nn.Module,
        regressor_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        tokenizer_max_length: Optional[int] = None,
        embedder_optimizer: Optional[torch.optim.Optimizer] = None,
        metadata_embedder: Optional[nn.Module] = None,
        metadata_embedder_output_dim: Optional[int] = None,
        metadata_projector: Optional[nn.Module] = None,
        metadata_projector_output_dim: Optional[int] = None,
        metadata_embedder_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.automatic_optimization = False

        self.tokenizer = tokenizer
        self.embedder = embedder
        self.embedder_output_dim = embedder_output_dim
        self.regressor = regressor

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
        # self.layer_norm = nn.LayerNorm(embedder_output_dim)

        self.criterion = nn.MSELoss()

        self.train_rank_corr = SpearmanCorrCoef()
        self.val_rank_corr = SpearmanCorrCoef()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_rank_corr_best = MaxMetric()

    def _emb_metadata(self, m: Tuple[str]) -> torch.Tensor:
        context = (
            torch.no_grad() if not self.optimize_metadata_embedder else nullcontext()
        )
        with context:
            encoded_input = self.tokenizer(
                m,
                max_length=self.hparams.tokenizer_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            emb_m = self.metadata_embedder(**encoded_input)
            emb_m = self._mean_pooling(emb_m, encoded_input["attention_mask"])
        return emb_m

    def forward(self, x: Tuple[str], m: Tuple[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(
            x,
            max_length=self.hparams.tokenizer_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        x_emb = self.embedder(**encoded_input)
        x_emb = self._mean_pooling(x_emb, encoded_input["attention_mask"])

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
        self.val_rank_corr.reset()
        self.val_rank_corr_best.reset()

    def model_step(
        self, batch: Tuple[Tuple[str], torch.Tensor, Tuple[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, m = batch
        preds = self.forward(x, m)
        loss = self.criterion(preds.squeeze(), y.squeeze())
        return loss, preds, y

    def training_step(
        self, batch: Tuple[Tuple[str], torch.Tensor, Tuple[str]], batch_idx: int
    ) -> torch.Tensor:
        if self.optimize_metadata_embedder:
            opt_regress, opt_embed, opt_m_embed = self.optimizers()
            opt_m_embed.zero_grad()
        else:
            opt_regress, opt_embed = self.optimizers()
        opt_regress.zero_grad()
        opt_embed.zero_grad()

        loss, preds, targets = self.model_step(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=0.5)
        
        if self.optimize_metadata_embedder:
            torch.nn.utils.clip_grad_norm_(self.metadata_embedder.parameters(), max_norm=0.5)

        opt_embed.step()
        opt_regress.step()
        if self.optimize_metadata_embedder:
            opt_m_embed.step()

        self.train_loss(loss)
        self.train_rank_corr(preds.squeeze(), targets.squeeze())
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/rank_corr",
            self.train_rank_corr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(
        self, batch: Tuple[Tuple[str], torch.Tensor, Tuple[str]], batch_idx: int
    ) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_rank_corr(preds.squeeze(), targets.squeeze())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/rank_corr",
            self.val_rank_corr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        rank_corr = self.val_rank_corr.compute()
        self.val_rank_corr_best(rank_corr)

        self.log(
            "val/rank_corr_best",
            self.val_rank_corr_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
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
