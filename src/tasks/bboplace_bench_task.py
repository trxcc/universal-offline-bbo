from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np

from src.tasks.base import OfflineBBOTask


class BBOPlacementTask(OfflineBBOTask):
    _support_benchmarks = (
        "adaptec1",
        "adaptec2",
        "adaptec3",
        "adaptec4",
        "bigblue1",
        "bigblue3",
    )

    def __init__(
        self,
        benchmark_name: str,
        root_dir: Path,
        data_preserved_ratio: float = 0.3,
        reevaluate: bool = False,
    ) -> None:
        assert benchmark_name in self._support_benchmarks

        import sys

        sys.path.append(str(root_dir / "thirdparty_benchmark" / "BBOPlace_miniBench"))
        sys.path.append(
            str(root_dir / "thirdparty_benchmark" / "BBOPlace_miniBench" / "src")
        )

        from thirdparty_benchmark.BBOPlace_miniBench.src.evaluator import Evaluator
        from thirdparty_benchmark.BBOPlace_miniBench.src.place_utils.args_parser import (
            parse_args,
        )

        args = SimpleNamespace(
            **{"placer": "gg", "benchmark": f"ispd2005/{benchmark_name}"}
        )
        args = parse_args(args)

        self.evaluator = Evaluator(args)
        self.evaluate_func = lambda x: self.evaluator.evaluate(x)[0] * (
            -1
        )  # Since maximization

        data_dir = (
            root_dir / "thirdparty_benchmark" / "BBOPlace_miniBench" / "bboplace_data"
        )
        x_all = np.load(f"{data_dir}/{benchmark_name}_pso_x.npy")
        y_all = np.load(f"{data_dir}/{benchmark_name}_pso_y.npy")
        if reevaluate:
            y_all = self.evaluate_func(x_all) * (-1)
            np.save(f"{data_dir}/{benchmark_name}_pso_y.npy", y_all)

        indices = np.argsort(y_all.flatten())[: int(len(x_all) * data_preserved_ratio)]
        task_x = x_all[indices]
        task_y = y_all[indices] * (-1)  # Since maximization
        # if reevaluate:
        #     task_y = self.evaluate_func(task_x)

        full_y_min, full_y_max = task_y.min(), task_y.max()

        self.benchmark_name = benchmark_name

        super(BBOPlacementTask, self).__init__(
            benchmark_name,
            task_type="Integer",
            x_np=task_x,
            y_np=task_y,
            full_y_min=full_y_min,
            full_y_max=full_y_max,
        )

    @property
    def eval_stability(self) -> bool:
        return False

    def _evaluate(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        x = x.reshape(-1, self.x_np.shape[1])
        return self.evaluate_func(x)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.evaluator.xl, self.evaluator.xu

    @property
    def ndim_problem(self) -> int:
        return self.x_np.shape[1]

    @property
    def num_classes(self) -> int:
        if self.task_type == "Continuous":
            raise ValueError("continuous task does not support num_classes attribute")
        return self.task.num_classes

    def test_evaluate(self, x):
        return self.evaluate_func(x).reshape(-1, 1)
