from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule, LightningDataModule
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers import T5ForConditionalGeneration
from transformers.tokenization_utils_base import BatchEncoding

from src.utils.io_utils import load_task_names


class OmnipredLatentModule(LightningModule):
    def __init__(
        self,
        model: Union[T5ForConditionalGeneration, nn.Module],
        optimizer: torch.optim.Optimizer,
        compile: bool,
        input_tokenizer: Any,
        output_tokenizer: Any,
        data_dir: Path,
        task_names: str,
        metadata_embedder: Optional[nn.Module] = None,
        metadata_embedder_output_dim: Optional[int] = None,
        numeric_interval: int = 50,
        scheduler=None,
        temperature: float = 0.07,
        non_shuffled_datamodule=None,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.model = model
        
        # 获取 encoder 输出维度
        self.hidden_size = self.model.config.hidden_size
        self.metadata_embedder = metadata_embedder

        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        
        # 添加投影头
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128),
        )
        
        self.metadata_projection_head = nn.Sequential(
            nn.Linear(metadata_embedder_output_dim, metadata_embedder_output_dim),
            nn.ReLU(),
            nn.Linear(metadata_embedder_output_dim, 128),
        )
        
        self.temperature = temperature
        self.non_shuffled_datamodule = non_shuffled_datamodule
        
        # Metrics
        self.train_total_loss = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_con_loss = MeanMetric()
        self.train_lip_loss = MeanMetric()
        
        self.init_weights()

    def init_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        self.projection_head.apply(init_weights)
        self.metadata_projection_head.apply(init_weights)

    def contrastive_loss(self, embeddings1, embeddings2):
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

    def lipschitz_loss(self, z, y):
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
            
        return ratio.mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get encoder outputs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Project encoder outputs for contrastive learning
        mean_pooled = self._mean_pooling(encoder_hidden_states, attention_mask)
        projected_embeddings = self.projection_head(mean_pooled)
        
        # Decoder forward pass
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     labels=labels,
        # )

        if labels is not None:
        # 使用完整的模型进行训练时的前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            decoder_outputs = outputs
        else:
        # 推理时的前向传播
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )
        
        return SimpleNamespace(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            encoder_hidden_states=encoder_hidden_states,
            projected_embeddings=projected_embeddings
        )

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _emb_metadata(self, m: Tuple[BatchEncoding]) -> torch.Tensor:
        with torch.no_grad():
            encoded_input = m.to(self.device)
            emb_m = self.metadata_embedder(**encoded_input)
            emb_m = emb_m.last_hidden_state
            emb_m = self._mean_pooling(emb_m, encoded_input["attention_mask"])
        return emb_m

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # 获取非打乱的batch用于Lipschitz约束
        try:
            non_shuffled_batch = next(self.non_shuffled_train_iter)
        except:
            self.non_shuffled_train_iter = iter(
                self.non_shuffled_datamodule.train_dataloader()
            )
            non_shuffled_batch = next(self.non_shuffled_train_iter)

        # 正常的前向传播
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )

        # 计算主任务损失
        main_loss = outputs.loss

        # 计算对比损失
        # 假设batch中包含metadata的embedding
        with torch.no_grad():
            m_embeddings = self._emb_metadata(batch["metadata"])
        metadata_embeddings = self.metadata_projection_head(m_embeddings)
        contrastive_loss = self.contrastive_loss(
            outputs.projected_embeddings, 
            metadata_embeddings
        )

        # 计算Lipschitz损失
        non_shuffled_outputs = self.model.encoder(
            input_ids=non_shuffled_batch["input_ids"].to(self.device),
            attention_mask=non_shuffled_batch["attention_mask"].to(self.device),
        ).last_hidden_state
        non_shuffled_emb = self._mean_pooling(non_shuffled_outputs, non_shuffled_batch["attention_mask"].to(self.device))
        
        task_data = {}
        for i in range(len(non_shuffled_batch["task_name"])):
            task = non_shuffled_batch["task_name"][i]
            if task not in task_data:
                task_data[task] = {"text": [], "value": []}

            task_data[task]["text"].append(non_shuffled_emb[i])
            if non_shuffled_batch["value"].dim() > 1:
                task_data[task]["value"].append(non_shuffled_batch["value"][i : i + 1])
            else:
                task_data[task]["value"].append(non_shuffled_batch["value"][i])

        lipschitz_loss = 0
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

            lipschitz_loss += self.lipschitz_loss(
                z=task_data[task]["text"].to(self.device), y=task_data[task]["value"].to(self.device)
            ) * (len(task_data[task]["value"]) / len(non_shuffled_batch["value"]))

        # 组合所有损失
        total_loss = main_loss + contrastive_loss / (contrastive_loss / main_loss).detach() + lipschitz_loss / (lipschitz_loss / main_loss).detach()

        # 记录各种损失
        self.train_total_loss(total_loss)
        self.train_loss(main_loss)
        self.train_con_loss(contrastive_loss)
        self.train_lip_loss(lipschitz_loss)

        # 日志记录
        self.log("train/main_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", self.train_total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/contrastive_loss", self.train_con_loss, on_step=True, on_epoch=True)
        self.log("train/lipschitz_loss", self.train_lip_loss, on_step=True, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self) -> None:
        pass
        # print("on_train_epoch_end")
        # if (self.current_epoch - self.last_numeric_epoch) >= self.numeric_interval:
        #     print("Computing numeric metrics")
        #     self.last_numeric_epoch = self.current_epoch
        #     train_loader = self.trainer.train_dataloader
        #     val_loader = self.trainer.val_dataloaders

        #     self.compute_numeric_metrics(train_loader, prefix="train")

        #     self.compute_numeric_metrics(val_loader, prefix="val")

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
            self.non_shuffled_datamodule.setup("fit")
            loader = self.non_shuffled_datamodule.train_dataloader()
            self.non_shuffled_train_iter = iter(loader)

            self.model = torch.compile(self.model)
            self.metadata_embedder = torch.compile(self.metadata_embedder)
            self.metadata_projection_head = torch.compile(self.metadata_projection_head)
            self.projection_head = torch.compile(self.projection_head)

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