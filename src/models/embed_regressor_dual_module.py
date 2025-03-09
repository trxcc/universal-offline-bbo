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
from src.models.components.mlp import SimpleMLP

log = RankedLogger(__name__, rank_zero_only=True)


class EmbedRegressorModule(LightningModule):

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
        embedder_optimizer: Optional[torch.optim.Optimizer] = None,
        metadata_embedder: Optional[nn.Module] = None,
        metadata_embedder_output_dim: Optional[int] = None,
        metadata_projector: Optional[nn.Module] = None,
        metadata_projector_output_dim: Optional[int] = None,
        metadata_embedder_optimizer: Optional[torch.optim.Optimizer] = None,
        cat_metadata: bool = False,
        from_pretrained: bool = True,
        finetune_embedder: bool = True,
        if_mean_pooling: bool = True,
        adapt_mlp_optimizer: Optional[torch.optim.Optimizer] = None,
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
        self.if_mean_pooling = if_mean_pooling
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

        if not self.if_mean_pooling:
            layers = []
            layers.append(nn.Linear(embedder_output_dim*512, 512))
            layers.append(nn.ReLU())
            self.adapt_mlp = nn.Sequential(*layers)

        self.criterion = nn.NLLLoss()

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

    def forward(self, x: Tuple[BatchEncoding], m: Tuple[BatchEncoding]) -> torch.Tensor:
        context = torch.no_grad() if not self.finetune_embedder else nullcontext()
        with context:
            encoded_input = x.to(self.device)
            x_emb = self.embedder(**encoded_input)
            if self.if_mean_pooling:
                x_emb = self._mean_pooling(x_emb, encoded_input["attention_mask"])
            else:
                B, L, D = x_emb[0].shape
                x_flat = x_emb[0].reshape(B, L*D)
                x_emb = self.adapt_mlp(x_flat)

        if self.has_metadata:
            m_emb = self._emb_metadata(m)
            m_emb = self.metadata_projector(m_emb)
            x_emb = torch.cat((x_emb, m_emb), dim=1)

        # mean, var, min_vals, max_vals = self.batch_statistics(x_emb)
    
        # print("均值:", mean)
        # print("方差:", var)
        # print("最小值:", min_vals)
        # print("最大值:", max_vals)
        # print("使用Z-score方法检测异常值:")
        # stats_zscore = self.detect_outliers(x_emb, method='zscore', threshold=3.0)
        # for dim, dim_stats in stats_zscore.items():
        #     if dim_stats['outliers_count'] > 0:
        #         print(f"\n{dim}:")
        #         print(f"异常值数量: {dim_stats['outliers_count']}")
        #         print(f"异常值: {dim_stats['outlier_values']}")
        #         print(f"异常值索引: {dim_stats['outlier_indices']}")
        #         print(f"均值: {dim_stats['mean']}")
        #         print(f"标准差: {dim_stats['std']}")
        # assert 0
        x_emb = self.batch_norm(x_emb)
        mean, std = self.regressor(x_emb)
        return mean, std
    
    def batch_statistics(self, vectors: torch.Tensor):
        mean = torch.mean(vectors, dim=0)
        var = torch.var(vectors, dim=0)
        min_vals, _ = torch.min(vectors, dim=0)
        max_vals, _ = torch.max(vectors, dim=0)
    
        return mean, var, min_vals, max_vals
    
    def detect_outliers(self, vectors: torch.Tensor, method='zscore', threshold=3.0):
        batch_size, dim = vectors.shape
        stats = {}
    
        for d in range(dim):
            dim_data = vectors[:, d]  # 提取第d维的所有数据
        
            if method == 'zscore':
            # 使用z-score方法
                mean = torch.mean(dim_data)
                std = torch.std(dim_data)
                z_scores = torch.abs((dim_data - mean) / std)
                outliers_mask = z_scores > threshold
                outlier_values = dim_data[outliers_mask]
            
                stats[f'dim_{d}'] = {
                'mean': mean.item(),
                'std': std.item(),
                'outliers_count': outliers_mask.sum().item(),
                'outlier_values': outlier_values.tolist(),
                'outlier_indices': outliers_mask.nonzero().flatten().tolist(),
                'min': torch.min(dim_data).item(),
                'max': torch.max(dim_data).item()
            }
            
            elif method == 'iqr':
            # 使用IQR方法
                q1 = torch.quantile(dim_data, 0.25)
                q3 = torch.quantile(dim_data, 0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
            
                outliers_mask = (dim_data < lower_bound) | (dim_data > upper_bound)
                outlier_values = dim_data[outliers_mask]
            
                stats[f'dim_{d}'] = {
                'q1': q1.item(),
                'q3': q3.item(),
                'iqr': iqr.item(),
                'lower_bound': lower_bound.item(),
                'upper_bound': upper_bound.item(),
                'outliers_count': outliers_mask.sum().item(),
                'outlier_values': outlier_values.tolist(),
                'outlier_indices': outliers_mask.nonzero().flatten().tolist(),
                'min': torch.min(dim_data).item(),
                'max': torch.max(dim_data).item()
            }
    
        return stats

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

        mean, std = self.forward(x, m)
        # loss = self.criterion(preds.squeeze(), y.squeeze())
        dist = torch.distributions.Normal(mean.squeeze(), std.squeeze())
        nll_loss = -dist.log_prob(y.squeeze()).mean()

        return nll_loss, mean, y, task_names

    def training_step(
        self,
        batch: Dict[str, Union[Tuple[BatchEncoding], torch.Tensor, str]],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.optimize_metadata_embedder:
            opt_regress, opt_embed, opt_m_embed = self.optimizers()
            opt_m_embed.zero_grad()
        elif self.finetune_embedder:
            opt_regress, opt_embed = self.optimizers()
            opt_embed.zero_grad()
        elif not self.if_mean_pooling:
            opt_regress, opt_adapter = self.optimizers()
        else:
            opt_regress = self.optimizers()
        opt_regress.zero_grad()

        loss, preds, targets, task_names = self.model_step(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=0.5)
        if not self.if_mean_pooling:
            torch.nn.utils.clip_grad_norm_(self.adapt_mlp.parameters(), max_norm=0.5)
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
        if not self.if_mean_pooling:
            opt_adapter.step()

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
            if not self.if_mean_pooling:
                self.adapt_mlp = torch.compile(self.adapt_mlp)

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

        if not self.if_mean_pooling:
            adapter_optimizer = self.hparams.adapt_mlp_optimizer(
                params=self.adapt_mlp.parameters()
            )
            optimizers.append({"optimizer": adapter_optimizer})

        if self.hparams.finetune_embedder:
            embedder_optimizer = self.hparams.embedder_optimizer(
                params=self.embedder.parameters()
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
                preds, _ = self.forward(x, m)
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
