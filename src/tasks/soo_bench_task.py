from typing import Dict, Optional, Tuple

import numpy as np
from soo_bench.Taskdata import OfflineTask, set_use_cache

from src.tasks.base import OfflineBBOTask


class SOOBenchTask(OfflineBBOTask):

    def __init__(
        self,
        task_name: str,
        benchmark_id: int,
        seed: int = 1,
        *,
        low: int = 0,
        high: int = 100,
        num_data: Optional[int] = None,
    ) -> None:
        assert low < high, f"low percentile must be lower than high percentile"
        set_use_cache(True)
        self.task = OfflineTask(task_name, benchmark=benchmark_id, seed=seed)
        task_desc = f"{task_name}_{benchmark_id}_{seed}"
        if num_data is None:
            num_data = (
                0  # will be reset to dimension of the problem * 1000 in SOO-Bench
            )

        self.task.sample_bound(num=num_data, low=low, high=high)
        task_x, task_y = self.task.x.copy(), self.task.y.copy()

        dic2y = np.load("src/tasks/dic2y_sb.npy", allow_pickle=True).item()
        full_y_min, full_y_max = dic2y[task_desc]

        self.task_type = "Continuous"

        super(SOOBenchTask, self).__init__(
            task_desc,
            x_np=task_x,
            y_np=task_y,
            full_y_min=full_y_min,
            full_y_max=full_y_max,
        )

    def evaluate(
        self, x: np.ndarray, return_normalized_y: bool = True
    ) -> Dict[str, np.ndarray]:
        if self.task_type == "Continuous":
            assert x.dtype in [
                np.float32,
                np.float64,
            ], f"Input dtype must be float32 or float64, but got {x.dtype}"
        elif self.task_type == "Categorical":
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
        score = self.task.predict(x)[0]
        score_dict = get_percentile_score(score)

        if return_normalized_y:
            normalized_score = (score - self.full_y_min) / (
                self.full_y_max - self.full_y_min
            )
            score_dict.update(
                get_percentile_score(normalized_score, prefix="normalized")
            )

        return score_dict

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.task.xl, self.task.xu

    @property
    def ndim_problem(self) -> int:
        return self.x_np.shape[1]

    @property
    def num_classes(self) -> int:
        if self.task_type == "Continuous":
            raise ValueError("continuous task does not support num_classes attribute")
        return self.task.num_classes

