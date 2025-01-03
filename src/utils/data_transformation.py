from typing import Callable, List

import numpy as np
import torch
from lightning import LightningModule

from src.tasks.base import OfflineBBOTask


@torch.no_grad()
def model_fitness_function_string(
    x: np.ndarray, m: str, model: LightningModule
) -> np.ndarray:
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    assert len(x.shape) == 1 or len(x.shape) == 2
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    batch_size, n_var = x.shape
    ms = tuple([m for _ in range(batch_size)])

    def sol2str(single_solution):
        res_str = ", ".join(
            f"x{i}: {val.item()}" for i, val in enumerate(single_solution)
        )
        return res_str

    x_str = [sol2str(x0) for x0 in x]
    y_np = model(x_str, ms).cpu().numpy()
    assert len(y_np) == batch_size

    return y_np


@torch.no_grad()
def model_fitness_function(
    x: np.ndarray, model: LightningModule, task: OfflineBBOTask
) -> np.ndarray:
    assert len(x.shape) == 1 or len(x.shape) == 2
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    batch_size, n_var = x.shape

    if task.task_type == "Categorical":
        x = task.task.to_logits(x)
        x = x.reshape(x.shape[0], -1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.from_numpy(x).to(device, dtype=torch.float32)
    model = model.to(device)
    y_np = model(x_torch).cpu().numpy()
    assert len(y_np) == batch_size

    return y_np
