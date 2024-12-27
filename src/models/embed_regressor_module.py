from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torch.utils.data import DataLoader
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from tqdm import tqdm
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
        embedder_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.automatic_optimization = False

        self.tokenizer = tokenizer
        self.embedder = embedder
        self.embedder_output_dim = embedder_output_dim
        self.regressor = regressor

        self.batch_norm = nn.BatchNorm1d(embedder_output_dim)
        # self.layer_norm = nn.LayerNorm(embedder_output_dim)

        self.criterion = nn.MSELoss()

        self.train_rank_corr = SpearmanCorrCoef()
        self.val_rank_corr = SpearmanCorrCoef()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_rank_corr_best = MaxMetric()

    def forward(self, x: Tuple[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(
            x, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        x_emb = self.embedder(**encoded_input)
        x_emb = self._mean_pooling(x_emb, encoded_input["attention_mask"])
        # x_emb = self.layer_norm(x_emb)
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
        self, batch: Tuple[BatchEncoding, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_input, y = batch
        preds = self.forward(encoded_input)
        loss = self.criterion(preds.squeeze(), y.squeeze())
        return loss, preds, y

    def training_step(
        self, batch: Tuple[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        opt_regress, opt_embed = self.optimizers()
        opt_regress.zero_grad()
        opt_embed.zero_grad()

        loss, preds, targets = self.model_step(batch)
        loss.backward()

        opt_embed.step()
        opt_regress.step()

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

    def validation_step(self, batch: Tuple[str, torch.Tensor], batch_idx: int) -> None:
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

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.regressor_optimizer(
            params=self.trainer.model.parameters()
        )
        embedder_optimizer = self.hparams.embedder_optimizer(
            params=self.embedder.parameters()
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return [
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/rank_corr",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                },
                {
                    "optimizer": embedder_optimizer,
                },
            ]
        return [optimizer, embedder_optimizer]
