import json
import os
from typing import Any, Optional

import torch
from components.blt_tokenizer import ByteTokenizer
from components.entropy_dataset import EntropyDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from src.utils.io_utils import load_task_names


class EntropyDataModule(LightningDataModule):

    def __init__(
        self,
        task_names: str,
        *,
        tokenizer: Any,
        tokenizer_max_length: int = 128,
        cat_metadata: bool = True,
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
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
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
            task_names = load_task_names(self.hparams.task_names, self.hparams.data_dir)

            x_values = []
            y_values = []
            metadatas = []
            task_names_list = []
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
                task_names_list.extend([task_name for _ in range(len(xs))])

            dataset = EntropyDataset(
                x_values,
                y_values,
                tokenizer=self.tokenizer,
                tokenizer_max_length=self.tokenizer_max_length,
                concat_metadata=self.hparams.cat_metadata,
                metadatas=metadatas,
                task_names=task_names_list,
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


if __name__ == "__main__":
    datamodule = EntropyDataModule(
        task_names="AntMorphology-Exact-v0,DKittyMorphology-Exact-v0,Superconductor-RandomForest-v0,TFBind8-Exact-v0,TFBind10-Exact-v0",
        data_dir="data/entropy",
        tokenizer=ByteTokenizer(),
        tokenizer_max_length=128,
    )
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:
        print(batch)
        assert False
