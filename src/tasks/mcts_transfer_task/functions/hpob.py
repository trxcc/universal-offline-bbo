import json
import os
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb


def load_summary(root_dir):
    file_name = "summary-stats.json"
    path = os.path.join(root_dir, "saved-surrogates", file_name)
    with open(path, "rb") as f:
        summary_stats = json.load(f)
    return summary_stats


class HPOBProblem:
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
        root_dir: Path,
        nthread: int = 4,
    ):
        assert search_space_id.isnumeric()
        assert dataset_id.isnumeric()
        # if root_dir.startswith('~'):
        #     root_dir = os.path.expanduser(root_dir)

        # load model
        self.summary_stats = load_summary(root_dir)
        self.surrogate_name = "surrogate-" + search_space_id + "-" + dataset_id
        self.bst_surrogate = xgb.Booster()
        self.bst_surrogate.load_model(
            os.path.join(root_dir, "saved-surrogates", self.surrogate_name + ".json")
        )
        self.bst_surrogate.set_param({"nthread": nthread})

        self.dim = self.bst_surrogate.num_features()
        self.lb = torch.zeros(self.dim)
        self.ub = torch.ones(self.dim)
        self.name = "HPOB_{}".format(search_space_id)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Inputs:
            X: Tensor [bs, dim]
        Outputs:
            Tensor [bs, 1]
        """
        X = torch.from_numpy(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        assert X.ndim == 2
        X_np = X.cpu().detach().numpy()
        Y = []
        dim = X_np.shape[-1]
        for x in X_np:
            x_q = xgb.DMatrix(x.reshape(-1, dim))
            y = self.bst_surrogate.predict(x_q)
            Y.append(y)
        return torch.from_numpy(np.array(Y)).reshape(-1, 1).to(X).detach().cpu().numpy()

    def info(self):
        return self.summary_stats[self.surrogate_name]

    def __call__(self, X):
        return self.forward(X)
