from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers.tokenization_utils_base import BatchEncoding

from src.utils import RankedLogger
from src.utils.io_utils import load_task_names

log = RankedLogger(__name__, rank_zero_only=True)


class EntropyModule(LightningModule):

    def __init__(
        self,
        model: nn.Module,
        model_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        data_dir: Path,
        task_names: List[str],
        cat_metadata: Optional[bool] = False,
        from_pretrained: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.automatic_optimization = False
        self.task_names = load_task_names(task_names, data_dir)

        self.model = model

        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: Tuple[BatchEncoding]) -> torch.Tensor:
        encoded_input = x.to(self.device)
        return self.model(encoded_input)

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def model_step(
        self, batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["text"]
        src = x[:, :-1]
        tgt = x[:, 1:]
        logits = self.forward(src)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return loss, logits, tgt

    def training_step(
        self,
        batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]],
        batch_idx: int,
    ) -> torch.Tensor:
        opt_model = self.optimizers()
        opt_model.zero_grad()

        loss, preds, targets = self.model_step(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        opt_model.step()

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
        pass

    def validation_step(
        self,
        batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]],
        batch_idx: int,
    ) -> None:
        loss, preds, targets = self.model_step(batch)
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
        pass

    def test_step(self) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.model_optimizer(params=self.model.parameters())

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
            optimizers = [optimizer]

        return optimizers
