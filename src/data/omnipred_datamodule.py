import json
import os
from typing import Any, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from src.data.components.omnipred_dataset import OmnipredDataset


class OmnipredDataModule(LightningDataModule):

    def __init__(
        self,
        task_names: str,
        *,
        input_tokenizer: Any,
        output_tokenizer: Any,
        max_length: int = 128,
        concat_metadata: bool = True,
        data_dir: str = "data/",
        val_ratio: float = 0.2,
        batch_size: int = 128,
        num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert os.path.exists(data_dir)
        assert 0 < val_ratio < 1

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.save_hyperparameters(logger=False)

        # TODO: More flexible setting of transforms
        self.x_transforms = []
        self.y_transforms = []

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )
            # if self.trainer.accelerator:
            #     self.device = self.trainer.accelerator.device

        if not self.data_train or not self.data_val:
            # TODO: Load data as a single function
            if "," in self.hparams.task_names:
                task_names = list(self.hparams.task_names.split(","))
            else:
                task_names = [self.hparams.task_names]

            x_values = []
            y_values = []
            metadatas = []
            for task_name in task_names:
                data_file = f"{self.hparams.data_dir}/{task_name}.json"
                assert os.path.exists(data_file)
                with open(data_file, "r") as f:
                    data = json.load(f)

                ys = [d["y"] for d in data]
                xs = [", ".join(d["x"]) for d in data]
                y_values.extend(ys)
                x_values.extend(xs)
                assert len(xs) == len(ys)

                metadata_file = f"{self.hparams.data_dir}/{task_name}.metadata"
                with open(metadata_file, "r") as f:
                    metadata = f.read()
                    metadatas.extend([metadata for _ in range(len(xs))])

            dataset = OmnipredDataset(
                x_data=x_values,
                y_data=y_values,
                input_tokenizer=self.hparams.input_tokenizer,
                output_tokenizer=self.hparams.output_tokenizer,
                concat_metadata=self.hparams.concat_metadata,
                metadatas=metadatas,
                max_length=self.hparams.max_length,
            )
            lengths = [
                len(x_values) - int(len(x_values) * self.hparams.val_ratio),
                int(len(x_values) * self.hparams.val_ratio),
            ]
            self.data_train, self.data_val = random_split(
                dataset=dataset, lengths=lengths
            )

    def train_dataloader(self) -> DataLoader[Any]:
        sampler = None
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                self.data_train,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
            )

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=(sampler is None),
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        sampler = None
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                self.data_val,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
            )

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
