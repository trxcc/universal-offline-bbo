from pathlib import Path
from typing import Tuple

from src.data2str.task_data import ContinuousTaskData, TaskData
from src.data2str.task_metadata import ContinuousTaskMetadata, TaskMetadata
from src.tasks.base import OfflineBBOTask
from src.tasks.mcts_transfer_task.func_task import BBOBTask


def create_task_bbob(
    task_name: str, data_dir: Path, func_seed: int
) -> Tuple[OfflineBBOTask, TaskMetadata, TaskData]:
    task_desc = f"{task_name}_{func_seed}"

    task = BBOBTask(task_name, data_dir, func_seed)

    xl, xu = task.bounds
    metadata = ContinuousTaskMetadata(
        input_dim=task.x_np.shape[1],
        bounds=[(l, u) for l, u in zip(xl, xu)],
        name=f"BBOB_{task_desc}",
        objective=f"Minimizing {task_name} function value",
        description=task.seed2md[func_seed]["metadata"] + f"Shift seed is {func_seed}.",
    )
    data = ContinuousTaskData(task.x_np)

    return task, metadata, data
