import abc
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

DESIGN_TYPE = Sequence[Union[int, float]]


class BaseTask:
    @abc.abstractmethod
    def evaluate(
        self, x: Union[Sequence[DESIGN_TYPE], np.ndarray]
    ) -> Dict[str, Union[Sequence[Union[int, float]], np.ndarray]]:
        pass


class OfflineBBOTask(BaseTask):
    def __init__(
        self,
        task_name: str,
        *,
        x_np: np.ndarray,
        y_np: np.ndarray,
        full_y_min: Optional[Union[float, np.ndarray]] = None,
        full_y_max: Optional[Union[float, np.ndarray]] = None,
        x_ood_np: Optional[np.ndarray] = None,
        y_ood_np: Optional[np.ndarray] = None,
    ) -> None:
        self.task_name = task_name
        self.x_np = x_np
        self.y_np = y_np

        self.full_y_min = full_y_min
        self.full_y_max = full_y_max

        self.x_ood_np = x_ood_np
        self.y_ood_np = y_ood_np

    @property
    @abc.abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    def ndim_problem(self) -> int:
        return self.x_np.shape[1]
