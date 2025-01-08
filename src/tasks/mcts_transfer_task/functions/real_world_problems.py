from functools import partial
from typing import Any, List, Optional

import numpy as np
import torch
from scipy.stats.qmc import Sobol
from torch import Tensor


def scale_from_unit_square_to_domain(X, domain):
    # X contains elements in unit square, stretch and translate them to lie domain
    return X * domain.ptp(axis=1) + domain[:, 0]


def scale_from_domain_to_unit_square(X, domain):
    # X contains elements in domain, translate and stretch them to lie in unit square
    return (X - domain[:, 0]) / domain.ptp(axis=1)


class RealWorldProblem:
    def __init__(
        self,
        search_space_id: str,
        dataset_id: str,
    ):
        self.search_space_id = search_space_id
        self.dataset_id = dataset_id

        if search_space_id in (
            "LunarLander",
            # 'PDE',
            # 'Optics',
            "RobotPush",
            "Rover",
        ):
            # if search_space_id in ('LunarLander', 'PDE', 'Optics'):
            if search_space_id in ("LunarLander"):
                if search_space_id == "LunarLander":
                    from src.tasks.mcts_transfer_task.functions.real_world_utils.lunar_lander import (
                        LunarLanderProblem,
                    )

                    func_cls = LunarLanderProblem
                # elif search_space_id == 'PDE':
                #     from problems.real_world_utils.pdes import PDEVar
                #     func_cls = PDEVar
                # elif search_space_id == 'Optics':
                #     from problems.real_world_utils.optics import Optics
                #     func_cls = Optics
                else:
                    raise ValueError
                self.func = func_cls()
                self.dim = self.func.dim
            elif search_space_id in ("RobotPush", "Rover"):
                if search_space_id == "RobotPush":
                    from src.tasks.mcts_transfer_task.functions.real_world_utils.push_function import (
                        PushReward,
                    )

                    self.func = PushReward()
                    self.dim = 14
                elif search_space_id == "Rover":
                    from src.tasks.mcts_transfer_task.functions.real_world_utils.rover_function import (
                        create_rover_problem,
                    )

                    self.func = create_rover_problem()
                    self.dim = 60
            else:
                raise ValueError
            # we normalize X in evaluate_true function within the problem
            # so the bound is [0, 1] here
            self.lb = torch.zeros(self.dim)
            self.ub = torch.ones(self.dim)

            # transform
            bound_translation = 0.1
            bound_scaling = 0.1
            params_domain = [
                [-bound_translation, bound_translation] for _ in range(self.dim)
            ]
            params_domain.append([1 - bound_scaling, 1 + bound_scaling])
            params_domain = np.array(params_domain)
            sobol = Sobol(self.dim + 1, seed=0)
            params = sobol.random(512)
            self.params = scale_from_unit_square_to_domain(params, params_domain)

            idx = int(self.dataset_id)
            self.t = self.params[idx, 0:-1]
            self.s = self.params[idx, -1]
        else:
            raise NotImplementedError
        self.name = search_space_id

    def __call__(self, X) -> Any:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(X)  # .to('cuda')
        if X.ndim == 1:
            X = X.reshape(1, -1)
        assert X.ndim == 2
        assert (X >= self.lb).all() and (X <= self.ub).all()
        if self.search_space_id in (
            "LunarLander",
            # "PDE",
            # "Optics",
        ):
            Y = self.s * self.func(X - self.t)
            return Y.reshape(-1, 1).to(X).cpu().numpy()
        elif self.search_space_id in ("RobotPush", "Rover"):
            X_np = X.cpu().detach().numpy()
            Y = []
            for x in X_np:
                y = self.s * self.func(x - self.t)
                Y.append(y)
            return torch.from_numpy(np.array(Y)).reshape(-1, 1).to(X).cpu().numpy()
        elif self.search_space_id == "Furuta":
            X_np = X.cpu().detach().numpy()
            Y = []
            for x in X_np:
                y = self.func(x)
                Y.append(y)
            return torch.from_numpy(np.array(Y)).reshape(-1, 1).to(X).cpu().numpy()
        else:
            raise ValueError

    def reset_task(self, dataset_id: str):
        self.dataset_id = dataset_id

        if self.search_space_id in (
            "LunarLander",
            # "PDE",
            # "Optics",
            "RobotPush",
            "Rover",
        ):
            idx = int(self.dataset_id)
            self.t = self.params[idx, 0:-1]
            self.s = self.params[idx, -1]
        else:
            raise ValueError
