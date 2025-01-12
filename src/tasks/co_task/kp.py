from typing import Tuple, Dict

import os
from pathlib import Path 
import torch
import numpy as np
from src.tasks.base import OfflineBBOTask

class KPTask(OfflineBBOTask):
    _support_sizes = (50, 100, 200)
    def __init__(self, problem_size: int, data_dir: Path, max_data_size: int = 2500):
        assert problem_size in self._support_sizes
        self.problem_size = problem_size
        
        data_dir = data_dir / "co_data" / f"bi_kp_{problem_size}"
        x_file = data_dir / f"bi_kp_{problem_size}-x-0.npy"
        y_file = data_dir / f"bi_kp_{problem_size}-y-0.npy"
        
        task_x = np.load(x_file)
        task_y = np.load(y_file)[:, 0]
        
        full_y_min = np.min(task_y)
        full_y_max = np.max(task_y)

        interval = np.arange(0, task_x.shape[0], task_x.shape[0] // max_data_size)
        indices = np.argsort(task_y.squeeze())[interval]
        task_x = task_x[indices]
        task_y = task_y[indices]
        
        self.task_type = "Permutation"
        
        super(KPTask, self).__init__(
            f"KP_{problem_size}",
            x_np=task_x,
            y_np=task_y,
            full_y_min=full_y_min,
            full_y_max=full_y_max,
        )

        self.problem_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"MOKP_problem_{problem_size}.pt")
        self.pomo_size = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_problems()
    
    def load_problems(self, aug_factor=1, problems=None):

        if problems is not None:
            self.problems = problems
        elif os.path.exists(self.problem_file):
            self.problems = torch.load(self.problem_file)
        else:
            from src.tasks.co_task.MOKProblemDef import get_random_problems
            self.problems = get_random_problems(1, self.problem_size)
            torch.save(f = self.problem_file, obj = self.problems)

        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            raise NotImplementedError
        
        self.problems = self.problems.to(self.device)
        
        
    def evaluate(
        self, x: np.ndarray, return_normalized_y: bool = True
    ) -> Dict[str, np.ndarray]:
        if self.task_type == "Continuous":
            assert x.dtype in [
                np.float32,
                np.float64,
            ], f"Input dtype must be float32 or float64, but got {x.dtype}"
        elif self.task_type in ["Categorical", "Integer", "Permutation"]:
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
        score = self._evaluate(x)
        score_dict = get_percentile_score(score)

        if return_normalized_y:
            normalized_score = (score - self.full_y_min) / (
                self.full_y_max - self.full_y_min
            )
            score_dict.update(
                get_percentile_score(normalized_score, prefix="normalized")
            )

        return score_dict

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        # Status
        ####################################
        
        x = torch.from_numpy(x).reshape(x.shape[0], 1, -1).to(self.device)
        self.batch_size = x.shape[0]

        if self.problems.shape[0] == 1:
            self.problems = self.problems.repeat(self.batch_size, 1, 1)
        
        # items_mat = self.items_and_a_dummy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size+1, 3)
        # gathering_index = x[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, 3)
        # selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)

        self.items_and_a_dummy = torch.Tensor(np.zeros((self.batch_size, self.problem_size+1, 3))).to(self.device)
        self.items_and_a_dummy[:, :self.problem_size, :] = self.problems
        self.item_data = self.items_and_a_dummy[:, :self.problem_size, :]

        if self.problem_size == 50:
            capacity = 12.5
        elif self.problem_size == 100:
            capacity = 25
        elif self.problem_size == 200:
            capacity = 25
        else:
            raise NotImplementedError
        self.capacity = (torch.Tensor(np.ones((self.batch_size, ))) * capacity).to(self.device)

        self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size, ))).to(self.device)
        # self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size, ))).to(self.device)

        self.finished = torch.BoolTensor(np.zeros((self.batch_size,))).to(self.device)

        for i in range(x.shape[2]):
            selected_item = x[:, :, i].reshape((-1, ))
            # selected_item_data = torch.gather(self.item_data, 1, selected_item)
            selected_item_data = torch.stack([self.item_data[i, selected_item[i], :] \
                                              for i in range(self.batch_size)]).to(self.device)

            if self.finished.all():
                break

            exceed_capacity = self.capacity - selected_item_data[:, 0] < 0
            self.finished[exceed_capacity] = True
            unfinished_batches = ~self.finished

            self.accumulated_value_obj1[unfinished_batches] += selected_item_data[unfinished_batches, 1]
            # self.accumulated_value_obj2[unfinished_batches] += selected_item_data[unfinished_batches, 2]
            self.capacity[unfinished_batches] -= selected_item_data[unfinished_batches, 0]

        # res = torch.stack([self.accumulated_value_obj1,
        #                    self.accumulated_value_obj2],
        #                    axis = 1) * (-1)
        res = self.accumulated_value_obj1.detach().cpu().numpy()
        return res 

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.ndim_problem), np.ones(self.ndim_problem)

    @property
    def ndim_problem(self) -> int:
        return self.problem_size

    @property
    def num_classes(self) -> int:
        raise ValueError("KP tasks do not support categorical inputs")
