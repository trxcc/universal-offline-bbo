from typing import Callable, Optional, Sequence, Union

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from src.searcher.base import BaseSearcher


class WrappedPSOProblem(Problem):
    def __init__(
        self,
        n_var: int,
        score_fn: Callable[[np.ndarray], np.ndarray],
        MAXIMIZE: bool = True,
        xl: Optional[Union[Sequence[float], np.ndarray]] = None,
        xu: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> None:
        self.score_fn = score_fn
        self.MAXIMIZE = MAXIMIZE
        super().__init__(
            n_var=n_var,
            n_obj=1,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        assert x.shape[1] == self.n_var
        score = self.score_fn(x)
        assert score.shape[0] == x.shape[0]
        assert score.shape[1] == self.n_obj
        if self.MAXIMIZE:
            score = (-1) * score
        out["F"] = score


class PSOSearcher(BaseSearcher):
    def __init__(self, n_gen: int, MAXIMIZE: bool = True, *args, **kwargs) -> None:
        self.n_gen = n_gen
        super(PSOSearcher, self).__init__(*args, **kwargs)
        xl, xu = self.task.bounds
        self.pymoo_problem = WrappedPSOProblem(
            n_var=self.task.ndim_problem,
            score_fn=self.score_fn,
            xl=xl,
            xu=xu,
            MAXIMIZE=MAXIMIZE,
        )
        self.pso = PSO(
            pop_size=self.num_solutions,
            sampling=self.get_initial_designs(
                x=self.task.x_np, y=self.task.y_np, k=self.num_solutions
            )[0],
            eliminate_duplicates=True,
        )

    def run(self) -> np.ndarray:
        res = minimize(
            problem=self.pymoo_problem,
            algorithm=self.pso,
            termination=("n_gen", self.n_gen),
            verbose=True,
        )
        return res.pop.get("X")
