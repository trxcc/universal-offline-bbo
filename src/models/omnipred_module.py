from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers import T5ForConditionalGeneration

from src.utils import RankedLogger
log = RankedLogger(name=__file__, rank_zero_only=True)

class OmnipredModule(LightningModule):
    def __init__(
        self,
        model: Union[T5ForConditionalGeneration, nn.Module],
        optimizer: torch.optim.Optimizer,
        compile: bool,
        scheduler = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.model = model

        self.train_rank_corr = SpearmanCorrCoef()
        self.val_rank_corr = SpearmanCorrCoef()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_rank_corr_best = MaxMetric()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_rank_corr.reset()
        self.val_rank_corr_best.reset()

    def model_step(
        self, batch: Tuple[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = preds.loss
        # log.info(f"step loss = {loss.item()}")
        return loss, preds, batch["labels"]

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        # TODO: calculate primary output for metric calculation
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
    
    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(
        self, batch: Tuple[Dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        # TODO: calculate primary output for metric calculation
        self.log(
            "val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True
        )
    
    def on_validation_epoch_end(self) -> None:
        pass 

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
