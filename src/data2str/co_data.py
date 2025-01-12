from pathlib import Path
from typing import Optional, Tuple

from src.data2str.task_data import PermutationTaskData, TaskData
from src.data2str.task_metadata import PermutationTaskMetadata, TaskMetadata
from src.tasks.base import OfflineBBOTask
from src.tasks.co_task.kp import KPTask
from src.tasks.co_task.tsp import TSPTask

TASKNAMES = [
    "KP_50",
    "KP_100",
    "KP_200",
    "TSP_20",
    "TSP_50",
    "TSP_100",
]

TEXT_DESCRIPTIONS = {
    "KP_50": {
        "name": "KP_50",
        "description": "Single-objective knapspack problem with 50 items",
        "objective": "to maximize values of selected items under the constraint that the sum of weights does not exceed a capacity",
    },
    "KP_100": {
        "name": "KP_100",
        "description": "Single-objective knapspack problem with 100 items",
        "objective": "to maximize values of selected items under the constraint that the sum of weights does not exceed a capacity",
    },
    "KP_200": {
        "name": "KP_200",
        "description": "Single-objective knapspack problem with 200 items",
        "objective": "to maximize values of selected items under the constraint that the sum of weights does not exceed a capacity",
    },
    "TSP_20": {
        "name": "TSP_20",
        "description": "Single-objective raveling salesman problem with 20 cities",
        "objective": "to minimize the total travel cost to travel over all cities",
    },
    "TSP_50": {
        "name": "TSP_50",
        "description": "Single-objective raveling salesman problem with 50 cities",
        "objective": "to minimize the total travel cost to travel over all cities",
    },
    "TSP_100": {
        "name": "TSP_100",
        "description": "Single-objective raveling salesman problem with 100 cities",
        "objective": "to minimize the total travel cost to travel over all cities",
    },
}


def create_task(
    task_name: str, data_dir: Path
) -> Tuple[OfflineBBOTask, TaskMetadata, TaskData]:
    assert task_name in TASKNAMES

    task_name, problem_size = task_name.split("_")
    if task_name == "KP":
        task = KPTask(problem_size=int(problem_size), data_dir=data_dir)
    elif task_name == "TSP":
        task = TSPTask(problem_size=int(problem_size), data_dir=data_dir)
    else:
        raise NotImplementedError

    metadata = PermutationTaskMetadata(
        input_dim=int(problem_size),
    )
    data = PermutationTaskData(task.x_np)

    return task, metadata, data
