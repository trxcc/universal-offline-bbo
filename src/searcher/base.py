import abc
import time
from typing import Callable, Tuple

import numpy as np
from pypop7.optimizers.core.optimizer import Optimizer as PypopOptimizer

from src.tasks.base import OfflineBBOTask


class BaseSearcher:
    def __init__(
        self,
        task: OfflineBBOTask,
        score_fn: Callable[[np.ndarray], np.ndarray],
        num_solutions: int,
    ) -> None:
        self.task = task
        self.score_fn = score_fn
        self.num_solutions = num_solutions

    @staticmethod
    def get_initial_designs(
        x: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.argsort(y.squeeze())[-k:]
        return x[indices], y[indices]

    @abc.abstractmethod
    def run(self) -> np.ndarray:
        pass


class WrappedPypopSearcher(PypopOptimizer, BaseSearcher):
    def __init__(
        self, problem: dict, options: dict, task: OfflineBBOTask, num_solutions: int
    ) -> None:

        PypopOptimizer.__init__(self, problem, options)
        BaseSearcher.__init__(
            self,
            task=task,
            score_fn=problem["fitness_function"],
            num_solutions=num_solutions,
        )
        self.best_x_pop = None
        self.best_y_pop = None

    def initialize(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_initial_designs(
            x=self.task.x_np, y=self.task.y_np, k=self.num_solutions
        )

    @abc.abstractmethod
    def iterate(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _evaluate_fitness(self, x: np.ndarray, args=None) -> float:
        self.start_function_evaluations = time.time()
        if args is None:
            y = self.fitness_function(x)
        else:
            y = self.fitness_function(x, args=args)
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1
        # update best-so-far solution (x) and fitness (y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), float(y)

        # Initialize or update best populations
        if self.best_x_pop is None:
            # First evaluation: initialize arrays
            self.best_x_pop = np.tile(x, (self.num_solutions, 1))
            self.best_y_pop = np.full(self.num_solutions, y)
        else:
            # Check if current solution should be added to the population
            max_idx = np.argmax(
                self.best_y_pop
            )  # Index of worst solution in population
            if y < self.best_y_pop[max_idx]:
                # Replace worst solution with current one
                self.best_x_pop[max_idx] = np.copy(x)
                self.best_y_pop[max_idx] = y

        # update all settings related to early stopping
        if (self._base_early_stopping - y) <= self.early_stopping_threshold:
            self._counter_early_stopping += 1
        else:
            self._counter_early_stopping, self._base_early_stopping = 0, y
        return float(y)
