import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize

from src.searcher.base import BaseSearcher
from src.searcher.pymoo_utils import WrappedPymooProblem


class PSOSearcher(BaseSearcher):
    def __init__(self, n_gen: int, MAXIMIZE: bool = True, *args, **kwargs) -> None:
        self.n_gen = n_gen
        super(PSOSearcher, self).__init__(*args, **kwargs)
        xl, xu = self.task.bounds
        self.pymoo_problem = WrappedPymooProblem(
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
