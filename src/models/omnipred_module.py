from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers import T5ForConditionalGeneration

from src.utils.io_utils import load_task_names


class OmnipredModule(LightningModule):
    def __init__(
        self,
        model: Union[T5ForConditionalGeneration, nn.Module],
        optimizer: torch.optim.Optimizer,
        compile: bool,
        input_tokenizer: Any,
        output_tokenizer: Any,
        data_dir: Path,
        task_names: str,
        numeric_interval: int = 50,
        scheduler=None,
        # finetune: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.model = model
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        self.task_names = load_task_names(task_names, data_dir)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_numeric_mse = MeanMetric()
        self.train_numeric_rank_corr = SpearmanCorrCoef()
        self.val_numeric_mse = MeanMetric()
        self.val_numeric_rank_corr = SpearmanCorrCoef()

        self.last_numeric_epoch = -1
        self.numeric_interval = 1
        # self.finetune = finetune

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # if self.finetune:
        #     return self.model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         # labels=labels,
        #     )
        # else:
        return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )

    def on_train_start(self) -> None:
        self.val_loss.reset()
        # self.val_rank_corr.reset()
        # self.val_rank_corr_best.reset()

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
        return loss, outputs, batch["labels"]

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # Update metrics
        self.train_loss(loss)

        # Log metrics
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="train/loss",
        )

        return loss

    def on_train_epoch_end(self) -> None:
        print("on_train_epoch_end")
        if (self.current_epoch - self.last_numeric_epoch) >= self.numeric_interval:
            print("Computing numeric metrics")
            self.last_numeric_epoch = self.current_epoch
            train_loader = self.trainer.train_dataloader
            val_loader = self.trainer.val_dataloaders

            self.compute_numeric_metrics(train_loader, prefix="train")

            self.compute_numeric_metrics(val_loader, prefix="val")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)

        # Log metrics
        self.log(
            "val/loss",
            self.val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            metric_attribute="val/loss",
        )

    def on_validation_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            # Calculate total steps
            total_steps = self.trainer.estimated_stepping_batches

            scheduler = self.hparams.scheduler(
                optimizer=optimizer, num_training_steps=total_steps
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
        **kwargs,
    ) -> torch.Tensor:
        generation_kwargs = {
            "decoder_start_token_id": self.output_tokenizer.bos_token_id,
            "bos_token_id": self.output_tokenizer.bos_token_id,
            "pad_token_id": self.output_tokenizer.pad_token_id,
            "eos_token_id": self.output_tokenizer.eos_token_id,
            "max_length": max_length,
            "num_beams": 4,
            "do_sample": False,
            "early_stopping": True,
        }

        generation_kwargs.update(kwargs)

        return self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            **generation_kwargs,
        )

    def generate_numbers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        **kwargs,
    ):
        predictions = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs,
        )

        pred_numbers = []
        for pred in predictions:
            try:
                num = float(
                    self.output_tokenizer.decode(
                        pred[pred != -100], skip_special_tokens=True
                    )
                )
                pred_numbers.append(num)
            except ValueError:
                pred_numbers.append(float("-inf"))

        # Return a two-dimensional vector
        return torch.tensor(pred_numbers, device=self.device).reshape(-1, 1)

    def compute_numeric_metrics(self, dataloader, prefix="val"):
        pass
        # self.eval()
        # numeric_mse = MeanMetric().to(self.device)
        # rank_corr = {
        #     task: SpearmanCorrCoef().to(self.device) for task in self.task_names
        # }

        # task_preds = {task: [] for task in self.task_names}
        # task_targets = {task: [] for task in self.task_names}
        # with torch.no_grad():
        #     for batch in dataloader:
        #         input_ids = batch["input_ids"].to(self.device)
        #         attention_mask = batch["attention_mask"].to(self.device)
        #         labels = batch["labels"].to(self.device)
        #         task_names = batch["task_name"]
        #         pred_numbers = self.generate_numbers(
        #             input_ids=input_ids, attention_mask=attention_mask
        #         )

        #         target_numbers = []
        #         for label in labels:
        #             try:
        #                 num = float(
        #                     self.output_tokenizer.decode(
        #                         label[label != -100], skip_special_tokens=True
        #                     )
        #                 )
        #                 target_numbers.append(num)
        #             except ValueError:
        #                 target_numbers.append(float("-inf"))

        #         target_numbers = torch.tensor(
        #             target_numbers, device=self.device
        #         ).reshape(-1, 1)

        #         numeric_mse(
        #             F.mse_loss(pred_numbers.squeeze(), target_numbers.squeeze())
        #         )

        #         for i, task_name in enumerate(task_names):
        #             task_preds[task_name].append(pred_numbers[i])
        #             task_targets[task_name].append(target_numbers[i])

        # self.log(
        #     f"{prefix}/numeric_mse",
        #     numeric_mse.compute(),
        #     on_epoch=True,
        #     prog_bar=True,
        #     metric_attribute=f"{prefix}/numeric_mse",
        # )
        # all_corrs = []
        # for task_name in self.task_names:
        #     task_pred = torch.stack(task_preds[task_name])
        #     task_target = torch.stack(task_targets[task_name])
        #     print(task_pred.shape, task_target.shape)
        #     task_rank_corr = rank_corr[task_name](
        #         task_pred.squeeze(), task_target.squeeze()
        #     )
        #     all_corrs.append(task_rank_corr)
        #     self.log(
        #         f"{prefix}/rank_corr_{task_name}",
        #         task_rank_corr,
        #         on_epoch=True,
        #         sync_dist=True,
        #         prog_bar=False,
        #         metric_attribute=f"{prefix}/rank_corr_{task_name}",
        #     )

        # avg_corr = torch.stack(all_corrs).mean()
        # self.log(
        #     f"{prefix}/rank_corr_avg",
        #     avg_corr,
        #     on_epoch=True,
        #     sync_dist=True,
        #     prog_bar=True,
        #     metric_attribute=f"{prefix}/rank_corr_avg",
        # )

        # self.train()
