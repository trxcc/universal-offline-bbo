from typing import Callable, List

import numpy as np
import torch
from lightning import LightningModule


@torch.no_grad()
def model_fitness_function(x: np.ndarray, model: LightningModule) -> np.ndarray:
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    assert len(x.shape) == 1 or len(x.shape) == 2
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    batch_size, n_var = x.shape

    def sol2str(single_solution):
        res_str = ", ".join(
            f"x{i}: {val.item()}" for i, val in enumerate(single_solution)
        )
        return res_str

    x_str = [sol2str(x0) for x0 in x]
    y_np = model(x_str).cpu().numpy()
    assert len(y_np) == batch_size

    return y_np
