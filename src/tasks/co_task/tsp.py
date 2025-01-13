import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from src.tasks.base import OfflineBBOTask
from src.tasks.co_task.MOTSProblemDef import (
    augment_xy_data_by_64_fold_2obj,
    get_random_problems,
)


class TSPTask(OfflineBBOTask):
    _support_sizes = (20, 50, 100)

    def __init__(self, problem_size: int, data_dir: Path, max_data_size: int = 2500):
        assert problem_size in self._support_sizes
        self.problem_size = problem_size

        data_dir = data_dir / "co_data" / f"bi_tsp_{problem_size}"
        x_file = data_dir / f"bi_tsp_{problem_size}-x-0.npy"
        y_file = data_dir / f"bi_tsp_{problem_size}-y-0.npy"

        task_x = np.load(x_file)
        task_y = np.load(y_file)[:, 0] * (-1)  # Since maximization

        full_y_min = np.min(task_y)
        full_y_max = np.max(task_y)

        interval = np.arange(0, task_x.shape[0], task_x.shape[0] // max_data_size)
        indices = np.argsort(task_y.squeeze())[interval]
        task_x = task_x[indices]
        task_y = task_y[indices]

        super(TSPTask, self).__init__(
            f"TSP_{problem_size}",
            task_type="Permutation",
            x_np=task_x,
            y_np=task_y,
            full_y_min=full_y_min,
            full_y_max=full_y_max,
        )

        self.problem_size = problem_size
        self.problem_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"MOTSP_problem_{problem_size}.pt",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_problems()

    @property
    def eval_stability(self) -> bool:
        return True

    def load_problems(self, aug_factor=1, problems=None):
        if problems is not None:
            self.problems = problems
        elif os.path.exists(self.problem_file):
            self.problems = torch.load(f=self.problem_file)
        else:
            self.problems = get_random_problems(1, self.problem_size)
            torch.save(obj=self.problems, f=self.problem_file)

        # problems.shape: (1, problem, 2)
        if aug_factor > 1:
            if aug_factor == 64:
                self.problems = augment_xy_data_by_64_fold_2obj(self.problems)
            else:
                raise NotImplementedError

        self.problems = self.problems.to(self.device)

    def evaluate(
        self, x: np.ndarray, return_normalized_y: bool = True
    ) -> Dict[str, np.ndarray]:
        if self.task_type == "Continuous":
            assert x.dtype in [
                np.float32,
                np.float64,
            ], f"Input dtype must be float32 or float64, but got {x.dtype}"
        elif self.task_type in ["Categorical", "Integer", "Permutation"]:
            assert x.dtype in [
                np.int32,
                np.int64,
            ], f"Input dtype must be int32 or int64, but got {x.dtype}"
        else:
            raise NotImplementedError

        def get_percentile_score(
            score: np.ndarray, prefix: str = ""
        ) -> Dict[str, float]:
            prefix = f"{prefix}/" if prefix != "" else prefix
            return {
                f"{prefix}score/100th": np.max(score).item(),
                f"{prefix}score/75th": np.percentile(score, 75).item(),
                f"{prefix}score/50th": np.median(score).item(),
                f"{prefix}score/25th": np.percentile(score, 25).item(),
            }

        x = x.reshape(-1, self.x_np.shape[1])
        score = self._evaluate(x)
        score_dict = get_percentile_score(score)

        if return_normalized_y:
            normalized_score = (score - self.full_y_min) / (
                self.full_y_max - self.full_y_min
            )
            score_dict.update(
                get_percentile_score(normalized_score, prefix="normalized")
            )

        return score_dict

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x).reshape((x.shape[0], 1, -1)).to(self.device)
        self.batch_size = x.shape[0]

        expanded_problems = self.problems.repeat(self.batch_size, 1, 1)

        gathering_index = x.unsqueeze(3).expand(
            self.batch_size, -1, self.problem_size, 4
        )
        # shape: (batch, 1, problem, 4)
        seq_expanded = expanded_problems[:, None, :, :].expand(
            self.batch_size, 1, self.problem_size, 4
        )

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, q, problem, 4)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths_obj1 = (
            ((ordered_seq[:, :, :, :2] - rolled_seq[:, :, :, :2]) ** 2).sum(3).sqrt()
        )
        # segment_lengths_obj2 = ((ordered_seq[:, :, :, 2:]-rolled_seq[:, :, :, 2:])**2).sum(3).sqrt()

        travel_distances_obj1 = segment_lengths_obj1.sum(2)
        # travel_distances_obj2 = segment_lengths_obj2.sum(2)

        # travel_distances_vec = torch.stack([travel_distances_obj1,travel_distances_obj2], axis = 2)\
        #     .reshape((self.batch_size, self.n_obj))

        # out["G"] = np.ones(self.batch_size)
        # for i, x_i in enumerate(x):
        #     if torch.equal(x_i.data.sort(1)[0], \
        #             torch.arange(x_i.size(1), out=x_i.data.new()).view(1, -1).expand_as(x_i)):
        #         out["G"][i] = -1
        return travel_distances_obj1.cpu().numpy() * (-1)  # Since minimization
        # shape: (batch, pomo)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.ndim_problem), np.ones(self.ndim_problem)

    @property
    def ndim_problem(self) -> int:
        return self.problem_size

    @property
    def num_classes(self) -> int:
        raise ValueError("TSP tasks do not support categorical inputs")
