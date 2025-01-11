from pathlib import Path
from typing import Optional, Tuple

from src.data2str.task_data import IntegerTaskData, TaskData
from src.data2str.task_metadata import IntegerTaskMetadata, TaskMetadata
from src.tasks.base import OfflineBBOTask
from src.tasks.bboplace_bench_task import BBOPlacementTask

TASKNAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4", "bigblue1", "bigblue3"]

CONTINUOUS_TASKS = [
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue3",
]

TEXT_DESCRIPTIONS = {
    name: {
        "name": name,
        "description": f"Chip placement on {name} in ISPD 2005",
        "objective": "to minimize the half-perimeter wirelength (HPWL) of macro placement",
    }
    for name in TASKNAMES
}


def create_task(
    benchmark_name: str, root_dir: Path
) -> Tuple[OfflineBBOTask, TaskMetadata, TaskData]:
    task_desc = f"{benchmark_name}"

    is_continuous = task_desc in CONTINUOUS_TASKS
    assert is_continuous, "Our work in BBOPlace-Bench only support continuous tasks"

    task = BBOPlacementTask(
        benchmark_name=benchmark_name,
        root_dir=root_dir,
    )

    xl, xu = task.bounds
    metadata = IntegerTaskMetadata(
        input_dim=task.x_np.shape[1],
        bounds=[(l, u) for l, u in zip(xl, xu)],
        **TEXT_DESCRIPTIONS[task_desc],
    )
    data = IntegerTaskData(task.x_np)

    return task, metadata, data
