from pathlib import Path
from typing import Tuple

from src.data2str.task_data import ContinuousTaskData, TaskData
from src.data2str.task_metadata import ContinuousTaskMetadata, TaskMetadata
from src.tasks.base import OfflineBBOTask
from src.tasks.mcts_transfer_task.func_task import BBOBTask, HPOBTask, RealWorldTask


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
        objective=f"Maximizing {task_name} function value",
        description=task.seed2md[func_seed]["metadata"] + f"Shift seed is {func_seed}.",
    )
    data = ContinuousTaskData(task.x_np)

    return task, metadata, data


REAL_WORLD_TEXTS = {
    "LunarLander": {
        "objective": "to maximize the mean terminal reward across a consistent batch of 50 randomly generated landscape",
        "description": "Learn the parameters of a controller for a lunar lander; ",
    },
    "RobotPush": {
        "objective": "to minimize he distance between a predefined target location and two objects",
        "description": "Control the robot to push items to a designated location; ",
    },
    "Rover": {
        "objective": "to design a reasonable trajectory to minimize the cost",
        "description": "2D trajectories optimization for a rover; ",
    },
}


def create_task_real_world(
    task_name: str, data_dir: Path, func_seed: int
) -> Tuple[OfflineBBOTask, TaskMetadata, TaskData]:
    task_desc = f"{task_name}_{func_seed}"

    task = RealWorldTask(task_name, data_dir, func_seed)

    xl, xu = task.bounds
    metadata = ContinuousTaskMetadata(
        input_dim=task.x_np.shape[1],
        bounds=[(l, u) for l, u in zip(xl, xu)],
        name=f"{task_desc}",
        objective=REAL_WORLD_TEXTS[task_name]["objective"],
        description=REAL_WORLD_TEXTS[task_name]["description"]
        + task.seed2md[func_seed]["metadata"]
        + f"Shift seed is {func_seed}.",
    )
    data = ContinuousTaskData(task.x_np)

    return task, metadata, data


def create_task_hpob(
    search_space_id: str, dataset_id: int, root_dir: Path, data_dir: Path
) -> Tuple[OfflineBBOTask, TaskMetadata, TaskData]:
    task_desc = f"HPOB_{search_space_id}"
    print(task_desc)
    task = HPOBTask(task_desc, dataset_id, root_dir, data_dir)

    xl, xu = task.bounds
    metadata = ContinuousTaskMetadata(
        input_dim=task.x_np.shape[1],
        bounds=[(l, u) for l, u in zip(xl, xu)],
        name=f"{task_desc}",
        objective="Minimization on HPO-B benchmark",
        description=task.did2md[dataset_id]["metadata"],
    )
    data = ContinuousTaskData(task.x_np)

    return task, metadata, data
