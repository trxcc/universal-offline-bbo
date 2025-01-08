import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.tasks.base import OfflineBBOTask
from src.tasks.mcts_transfer_task.functions.bbob import (
    GriewankRosenbrock,
    Lunacek,
    Rastrigin,
    RosenbrockRotated,
    SharpRidge,
)
from src.tasks.mcts_transfer_task.functions.real_world_problems import RealWorldProblem
from src.tasks.mcts_transfer_task.utils import load_mcts_transfer_data


class BBOBTask(OfflineBBOTask):
    _name2func = {
        "GriewankRosenbrock": GriewankRosenbrock,
        "Lunacek": Lunacek,
        "Rastrigin": Rastrigin,
        "RosenbrockRotated": RosenbrockRotated,
        "SharpRidge": SharpRidge,
    }

    def __init__(self, task_name: str, data_dir: Path, func_seed: int = 0) -> None:
        assert task_name in self._name2func.keys()
        self.data = load_mcts_transfer_data(data_dir, "bbob")[task_name]

        self.seed2md = {}
        for dataset_id, dataset in self.data.items():
            collect_algo, collect_seed, eval_seed = tuple(dataset_id.split("+"))
            self.seed2md[eval(eval_seed)] = {
                "metadata": f"Collect algorithm: {collect_algo}, "
                f"with random seed = {collect_seed}. ",
                "X": np.array(dataset["X"]),
                "y": np.array(dataset["y"]),
            }

        func_type = self._name2func[task_name]()
        self.eval_function = lambda x: func_type(x, seed=func_seed) * (
            -1
        )  # since BBOB function is minimization
        # Besides, the data provide in mcts-transfer is maximizing

        self.seed_in_data = func_seed in self.seed2md.keys()
        if not self.seed_in_data:
            warnings.warn(
                f"Not support function seed in {task_name}. "
                "The search procedure will initialize with random designs. "
            )
            data_size = list(self.data.values())[0]["X"].shape[0]
            lower_bound, upper_bound = self.bound
            task_x = lower_bound + (upper_bound - lower_bound) * np.random.random(
                (data_size, self.ndim_problem)
            )
            task_y = np.zeros(shape=(data_size, 1))
        else:
            task_x = self.seed2md[func_seed]["X"]
            task_y = self.seed2md[func_seed]["y"]

        self.task_type = "Continuous"

        super(BBOBTask, self).__init__(
            f"{task_name}_{func_seed}",
            x_np=task_x,
            y_np=task_y,
            full_y_min=np.min(task_y) if self.seed_in_data else 0,
            full_y_max=np.max(task_y) if self.seed_in_data else 1,
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
        score = self.eval_function(x)
        score_dict = get_percentile_score(score)

        if return_normalized_y:
            if not self.seed_in_data:
                warnings.warn(
                    f"Not support function seed in {self.task_name}. "
                    "Only return unnormalized y."
                )
                return score_dict

            normalized_score = (score - self.full_y_min) / (
                self.full_y_max - self.full_y_min
            )
            score_dict.update(
                get_percentile_score(normalized_score, prefix="normalized")
            )

        return score_dict

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.full(self.ndim_problem, -5), np.full(self.ndim_problem, 5)

    @property
    def ndim_problem(self) -> int:
        return 10

    @property
    def num_classes(self) -> int:
        raise ValueError("BBOB tasks do not support categorical inputs")


class RealWorldTask(OfflineBBOTask):
    _task2dims = {
        "LunarLander": 12,
        "RobotPush": 14,
        "Rover": 60,
    }

    def __init__(
        self,
        task_name: str,
        data_dir: Path,
        func_seed: int = 0,
        reevaluate: bool = True,
    ) -> None:
        assert task_name in self._task2dims.keys()
        self.dim = self._task2dims[task_name]
        if task_name in ["RobotPush", "Rover"]:
            reevaluate = False

        self.data = load_mcts_transfer_data(data_dir, "real_world")[task_name]
        self.seed2md = {}
        for dataset_id, dataset in self.data.items():
            collect_algo, collect_seed, eval_seed = tuple(dataset_id.split("+"))
            self.seed2md[eval(eval_seed)] = {
                "metadata": f"Collect algorithm: {collect_algo}, "
                f"with random seed = {collect_seed}. ",
                "X": np.array(dataset["X"]),
                "y": np.array(dataset["y"]),
            }

        func_type = RealWorldProblem(task_name, str(func_seed))
        self.eval_function = lambda x: func_type(x) * (
            -1
        )  # since function in mcts-transfer is minimization
        # Besides, the data provide in mcts-transfer is maximizing

        self.seed_in_data = func_seed in self.seed2md.keys()
        task_x = self.seed2md[func_seed]["X"]
        task_y = self.seed2md[func_seed]["y"]
        if reevaluate:
            preds = np.array([self.eval_function(x0) for x0 in task_x])
            task_y = preds.squeeze()
            self.seed2md[func_seed]["y"] = preds

        self.task_type = "Continuous"

        super(RealWorldTask, self).__init__(
            f"{task_name}_{func_seed}",
            x_np=task_x,
            y_np=task_y,
            full_y_min=np.min(task_y) if self.seed_in_data else 0,
            full_y_max=np.max(task_y) if self.seed_in_data else 1,
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
        score = self.eval_function(x)
        score_dict = get_percentile_score(score)

        if return_normalized_y:
            if not self.seed_in_data:
                warnings.warn(
                    f"Not support function seed in {self.task_name}. "
                    "Only return unnormalized y."
                )
                return score_dict

            normalized_score = (score - self.full_y_min) / (
                self.full_y_max - self.full_y_min
            )
            score_dict.update(
                get_percentile_score(normalized_score, prefix="normalized")
            )

        return score_dict

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.ndim_problem), np.ones(self.ndim_problem)

    @property
    def ndim_problem(self) -> int:
        return self.dim

    @property
    def num_classes(self) -> int:
        raise ValueError("Real world tasks do not support categorical inputs")

    def test_evaluate(self, x):
        x = x.reshape(-1, self.x_np.shape[1])
        score = self.eval_function(x)
        return score
