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
        input_tokenizer: Any,
        output_tokenizer: Any,
        scheduler=None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.model = model
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        self.train_rank_corr = SpearmanCorrCoef()
        self.val_rank_corr = SpearmanCorrCoef()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_rank_corr_best = MaxMetric()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_rank_corr.reset()
        self.val_rank_corr_best.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss

        # predictions = self.generate(
        #     input_ids=batch["input_ids"],
        #     attention_mask=batch["attention_mask"],
        # )

        # pred_numbers = []
        # for pred in predictions:
        #     try:
        #         num = float(
        #             self.output_tokenizer.decode(pred, skip_special_tokens=True)
        #         )
        #         pred_numbers.append(num)
        #     except ValueError:
        #         pred_numbers.append(float("-inf"))

        # pred_numbers = torch.tensor(pred_numbers, device=self.device)

        # target_numbers = []
        # for label in batch["labels"]:
        #     try:
        #         num = float(
        #             self.output_tokenizer.decode(
        #                 label[label != -100], skip_special_tokens=True
        #             )
        #         )
        #         target_numbers.append(num)
        #     except ValueError:
        #         target_numbers.append(float("-inf"))

        # target_numbers = torch.tensor(target_numbers, device=self.device)

        # return loss, pred_numbers, target_numbers
        return loss, outputs, batch['labels']

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)
        # self.train_rank_corr(preds, targets)

        # Log metrics
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        # self.log(
        #     "train/rank_corr",
        #     self.train_rank_corr,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)
        # self.val_rank_corr(preds, targets)

        # Log metrics
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log(
        #     "val/rank_corr",
        #     self.val_rank_corr,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

    def on_validation_epoch_end(self) -> None:
        pass
        # rank_corr = self.val_rank_corr.compute()
        # self.val_rank_corr_best(rank_corr)
        # self.log("val/rank_corr_best", self.val_rank_corr_best.compute(), prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        if self.hparams.scheduler is not None:
            # Calculate total steps
            total_steps = self.trainer.estimated_stepping_batches
            
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                num_training_steps=total_steps
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Update lr every step instead of epoch
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        **kwargs
    ) -> torch.Tensor:
        generation_kwargs = {
            "decoder_start_token_id": self.output_tokenizer.bos_token_id,
            "bos_token_id": self.output_tokenizer.bos_token_id,
            "pad_token_id": self.output_tokenizer.pad_token_id,
            "eos_token_id": self.output_tokenizer.eos_token_id,
            "max_length": max_length,
            "num_beams": 4,
            "do_sample": False,
            "early_stopping": True
        }
        
        generation_kwargs.update(kwargs)
        
        return self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            **generation_kwargs
        )
