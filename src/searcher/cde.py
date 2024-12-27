""" not supported now, since pypop7 does not support batch evaluation
"""

from typing import Tuple

import numpy as np
from pypop7.optimizers.de.cde import CDE

from src.searcher.base import WrappedPypopSearcher
from src.tasks.base import OfflineBBOTask


class CDESearcher(CDE, WrappedPypopSearcher):
    def __init__(
        self,
        problem: dict,
        options: dict,
        task: OfflineBBOTask,
        num_solutions: int,
        MAXIMIZE: bool = True,
    ) -> None:

        self.MAXIMIZE = MAXIMIZE
        if self.MAXIMIZE:
            score_fn = problem["fitness_function"]
            problem["fitness_function"] = lambda x: (-1) * score_fn(x)
        options["n_individuals"] = num_solutions

        WrappedPypopSearcher.__init__(
            self,
            problem=problem,
            options=options,
            task=task,
            num_solutions=num_solutions,
        )
        CDE.__init__(self, problem, options)

    def initialize(self, args=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = WrappedPypopSearcher.initialize(self)
        v = np.empty((self.n_individuals, self.ndim_problem))
        return x, y, v

    def iterate(self, x=None, y=None, v=None, args=None):
        return CDE.iterate(self, x, y, v, args)

    def run(self) -> np.ndarray:
        self.optimize()
        return self.best_x_pop


if __name__ == "__main__":
    from src.tasks.design_bench_task import DesignBenchTask

    task = DesignBenchTask("Superconductor-RandomForest-v0")
    ndim_problem = task.x_np.shape[1]
    lb = task.x_np.min(axis=0)
    ub = task.x_np.max(axis=0)
    searcher = CDESearcher(
        problem={
            "fitness_function": task.evaluate,
            "ndim_problem": ndim_problem,
            "lower_boundary": lb,
            "upper_boundary": ub,
        },
        options={
            "max_function_evaluations": 5000,
            "n_individuals": 128,
        },
        task=task,
        num_solutions=128,
    )
    searcher.run()
