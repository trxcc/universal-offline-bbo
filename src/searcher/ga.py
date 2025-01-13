import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize

from src.searcher.base import BaseSearcher
from src.searcher.pymoo_utils import (
    RandomReplacementMutation,
    RoundingRepair,
    StartFromZeroRepair,
    UniformCrossover,
    WrappedPymooProblem,
)


class GASearcher(BaseSearcher):
    def __init__(
        self,
        n_gen: int,
        MAXIMIZE: bool = True,
        EVAL_STABILITY: bool = False,
        *args,
        **kwargs
    ) -> None:
        self.n_gen = n_gen
        super(GASearcher, self).__init__(*args, **kwargs)
        xl, xu = self.task.bounds
        self.EVAL_STABILITY = EVAL_STABILITY

        if self.task.task_type == "Categorical":
            operator = {
                "crossover": UniformCrossover(),
                "mutation": RandomReplacementMutation(),
            }
        elif self.task.task_type == "Permutation":
            operator = {
                "crossover": OrderCrossover(),
                "mutation": InversionMutation(),
            }
            if "TSP" in self.task.task_name:
                operator["repair"] = StartFromZeroRepair()
        elif self.task.task_type == "Integer":
            operator = {
                "repair": RoundingRepair(),
            }
        else:
            operator = {}

        if self.EVAL_STABILITY:
            self.X_all = []

            def record_callback(algorithm):
                X = algorithm.pop.get("X")
                self.X_all.append(X)

            operator["callback"] = record_callback

        self.pymoo_problem = WrappedPymooProblem(
            n_var=self.task.ndim_problem,
            score_fn=self.score_fn,
            xl=xl,
            xu=xu,
            MAXIMIZE=MAXIMIZE,
        )
        self.ga = GA(
            pop_size=self.num_solutions,
            sampling=self.get_initial_designs(
                x=self.task.x_np, y=self.task.y_np, k=self.num_solutions
            )[0],
            eliminate_duplicates=True,
            **operator
        )

    def run(self) -> np.ndarray:
        res = minimize(
            problem=self.pymoo_problem,
            algorithm=self.ga,
            termination=("n_gen", self.n_gen),
            verbose=True,
        )
        return res.pop.get("X")
