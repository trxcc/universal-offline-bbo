from typing import Tuple

import design_bench as db
from design_bench.task import Task

from .task_data import CategoricalTaskData, ContinuousTaskData, TaskData
from .task_metadata import CategoricalTaskMetadata, ContinuousTaskMetadata, TaskMetadata

TASKNAMES = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
]

CONTINUOUS_TASKS = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
]

CATEGORICAL_TASKS = [
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
]


def create_task(task_name: str) -> Tuple[Task, TaskMetadata, TaskData]:
    assert task_name in TASKNAMES

    # TODO: fix flags here
    is_continuous = task_name in CONTINUOUS_TASKS
    is_categorical = not is_continuous

    task: Task = db.make(task_name)
    if is_continuous:
        metadata = ContinuousTaskMetadata(
            name=task_name,
            input_dim=task.x.shape[1],
            bounds=[(float("-inf"), float("inf")) for _ in range(task.x.shape[1])],
        )
        data = ContinuousTaskData(task.x)

    elif is_categorical:
        metadata = CategoricalTaskMetadata(
            name=task_name,
            input_dim=task.x.shape[1],
            n_categories=task.num_classes,
        )
        data = CategoricalTaskData(task.x)

    else:
        raise NotImplementedError("Not supported")

    return task, metadata, data


if __name__ == "__main__":
    task, metadata, data = create_task("AntMorphology-Exact-v0")
    print(task, metadata, data)
    string_data = data.to_string()
    print(string_data[0])
