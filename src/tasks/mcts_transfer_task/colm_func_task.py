import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.tasks.base import OfflineBBOTask
from src.tasks.mcts_transfer_task.functions.colm_bbob import (
    Sphere,
    Rastrigin,
    BuecheRastrigin,
    LinearSlope,
    AttractiveSector,
    StepEllipsoidal,
    RosenbrockRotated,
    Ellipsoidal,
    Discus,
    BentCigar,
    SharpRidge,
    DifferentPowers,
    Weierstrass,
    SchaffersF7,
    SchaffersF7IllConditioned,
    GriewankRosenbrock,
    Schwefel,
    Katsuura,
    Lunacek,
    Gallagher101Me,
    Gallagher21Me,
    NegativeSphere,
    NegativeMinDifference,
    FonsecaFleming,
)
from src.tasks.mcts_transfer_task.functions.hpob import HPOBProblem
from src.tasks.mcts_transfer_task.functions.real_world_problems import RealWorldProblem
from src.tasks.mcts_transfer_task.colm_utils import load_mcts_transfer_data


class BBOBTask(OfflineBBOTask):
    _name2func = {
    "Sphere": Sphere,
    "Rastrigin": Rastrigin,
    "BuecheRastrigin": BuecheRastrigin,
    "LinearSlope": LinearSlope,
    "AttractiveSector": AttractiveSector,
    "StepEllipsoidal": StepEllipsoidal,
    "RosenbrockRotated": RosenbrockRotated,
    "Ellipsoidal": Ellipsoidal,
    "Discus": Discus,
    "BentCigar": BentCigar,
    "SharpRidge": SharpRidge,
    "DifferentPowers": DifferentPowers,
    "Weierstrass": Weierstrass,
    "SchaffersF7": SchaffersF7,
    "SchaffersF7IllConditioned": SchaffersF7IllConditioned,
    "GriewankRosenbrock": GriewankRosenbrock,
    "Schwefel": Schwefel,
    "Katsuura": Katsuura,
    "Lunacek": Lunacek,
    "Gallagher101Me": Gallagher101Me,
    "Gallagher21Me": Gallagher21Me,
    "NegativeSphere": NegativeSphere,
    "NegativeMinDifference": NegativeMinDifference,
    "FonsecaFleming": FonsecaFleming,
}

    def __init__(self, task_name: str, data_dir: Path, func_seed: int = 0, dim: int = 4) -> None:
        assert task_name in self._name2func.keys()
        self.data = load_mcts_transfer_data(data_dir, "bbob")[task_name]

        self.seed2md = {}
        for dataset_id, dataset in self.data.items():
            collect_algo, collect_seed, eval_seed = tuple(dataset_id.split("+"))
            self.seed2md[eval(eval_seed)] = {
                "metadata": f"Collect algorithm: {collect_algo}, "
                f"with random seed = {collect_seed}; ",
                "X": np.array(dataset["X"]),
                "y": np.array(dataset["Y"]),
            }

        func_type = self._name2func[task_name](dim=dim)
        self.eval_function = lambda x: func_type(x, seed=0) * (
            -1
        )  # since BBOB function is minimization
        # Besides, the data provide in mcts-transfer is maximizing

        self.seed_in_data = int(func_seed) in self.seed2md.keys()
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

        super(BBOBTask, self).__init__(
            f"{task_name}_{func_seed}",
            task_type="Continuous",
            x_np=task_x,
            y_np=task_y,
            full_y_min=np.min(task_y) if self.seed_in_data else 0,
            full_y_max=np.max(task_y) if self.seed_in_data else 1,
        )

    @property
    def eval_stability(self) -> bool:
        return True

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.eval_function(x0.squeeze()) for x0 in x])

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.full(self.ndim_problem, -5), np.full(self.ndim_problem, 5)

    @property
    def ndim_problem(self) -> int:
        return 4

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
                f"with random seed = {collect_seed}; ",
                "X": np.array(dataset["X"]),
                "y": np.array(dataset["y"]),
            }

        func_type = RealWorldProblem(task_name, str(func_seed))
        self.eval_function = lambda x: func_type(x) * (
            -1
        )  # since function in mcts-transfer is minimization
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
            if reevaluate:
                preds = np.array([self.eval_function(x0) for x0 in task_x])
                task_y = preds.squeeze()
                self.seed2md[func_seed]["y"] = preds

        super(RealWorldTask, self).__init__(
            f"{task_name}_{func_seed}",
            task_type="Continuous",
            x_np=task_x,
            y_np=task_y,
            full_y_min=np.min(task_y) if self.seed_in_data else 0,
            full_y_max=np.max(task_y) if self.seed_in_data else 1,
        )

    @property
    def eval_stability(self) -> bool:
        return False

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, self.x_np.shape[1])
        return self.eval_function(x)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.ndim_problem), np.ones(self.ndim_problem)

    @property
    def ndim_problem(self) -> int:
        return self.dim

    @property
    def num_classes(self) -> int:
        raise ValueError("Real world tasks do not support categorical inputs")


class HPOBTask(OfflineBBOTask):

    def __init__(
        self,
        task_name: str,
        dataset_id: str,
        root_dir: Path,
        data_dir: Path,
    ) -> None:
        search_space_id = task_name.split("_")[1]
        # assert search_space_id in self._id2dim.keys()
        # self.dim = self._id2dim[search_space_id]

        self.data = load_mcts_transfer_data(data_dir, "hpob-data")[search_space_id]
        self.did2md = {}
        for did, dataset in self.data.items():
            self.did2md[did] = {
                "metadata": f"HPOB algorithm {search_space_id} on dataset {did}",
                "X": np.array(dataset["X"]),
                "y": np.array(dataset["y"]).squeeze() * (-1),
            }  # For maximization

        self.problem = HPOBProblem(search_space_id, dataset_id, data_dir)
        self.eval_function = lambda x: self.problem(x) * (-1)  # For maximization

        self.did_in_data = dataset_id in self.did2md.keys()
        if not self.did_in_data:
            warnings.warn(
                f"Not support function dataset id in {task_name}. "
                "The search procedure will initialize with random designs. "
            )
            data_size = list(self.data.values())[0]["X"].shape[0]
            lower_bound, upper_bound = self.bound
            task_x = lower_bound + (upper_bound - lower_bound) * np.random.random(
                (data_size, self.ndim_problem)
            )
            task_y = np.zeros(shape=(data_size, 1))
        else:
            task_x = self.did2md[dataset_id]["X"]
            task_y = self.did2md[dataset_id]["y"]

        super(HPOBTask, self).__init__(
            f"HPOB_{search_space_id}",
            task_type="Continuous",
            x_np=task_x,
            y_np=task_y,
            full_y_min=np.min(task_y) if self.did_in_data else 0,
            full_y_max=np.max(task_y) if self.did_in_data else 1,
        )

    @property
    def eval_stability(self) -> bool:
        return False

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, self.x_np.shape[1])
        return self.eval_function(x)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.ndim_problem), np.ones(self.ndim_problem)

    @property
    def ndim_problem(self) -> int:
        return self.problem.dim

    @property
    def num_classes(self) -> int:
        raise ValueError("Real world tasks do not support categorical inputs")
