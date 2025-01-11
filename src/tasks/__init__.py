from pathlib import Path
from typing import List

from src.tasks.base import OfflineBBOTask

DESIGN_BENCH_TASKS = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    # Below are tasks that are not usually used in Design-Bench
    "HopperController-Exact-v0",
]

SOO_BENCH_TASKS = [
    "gtopx_data_2_1",
    "gtopx_data_3_1",
    "gtopx_data_4_1",
    "gtopx_data_6_1",
]

BBOPLACE_BENCH_TASKS = [
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue3",
]

REAL_WORLD_TASKS_PREFIX = ["LunarLander", "RobotPush", "Rover"]


def get_tasks(task_names: List[str], root_dir: Path) -> List[OfflineBBOTask]:
    tasks = []
    for task_entry in task_names:
        try:
            if task_entry in DESIGN_BENCH_TASKS:
                from src.tasks.design_bench_task import DesignBenchTask

                tasks.append(DesignBenchTask(task_name=task_entry, scale_up_ratio=2.0))
            elif task_entry in SOO_BENCH_TASKS:
                from src.tasks.soo_bench_task import SOOBenchTask

                task_name = task_entry[:10]
                benchmark_id, seed = task_entry[11:].split("_")
                benchmark_id = int(benchmark_id)
                seed = int(seed)
                tasks.append(
                    SOOBenchTask(
                        task_name=task_name,
                        benchmark_id=benchmark_id,
                        seed=seed,
                        low=25,
                        high=75,
                    )
                )
            elif task_entry in BBOPLACE_BENCH_TASKS:
                from src.tasks.bboplace_bench_task import BBOPlacementTask

                tasks.append(
                    BBOPlacementTask(benchmark_name=task_name, root_dir=root_dir)
                )
            elif task_entry.startswith(tuple(REAL_WORLD_TASKS_PREFIX)):
                from src.tasks.mcts_transfer_task.func_task import RealWorldTask

                task_name, seed = task_entry.split("_")
                tasks.append(
                    RealWorldTask(
                        task_name=task_name,
                        data_dir=root_dir / "data",
                        func_seed=int(seed),
                    )
                )
            elif task_entry.startswith("HPOB"):
                from src.tasks.mcts_transfer_task.func_task import HPOBTask

                task_name, search_space_id, dataset_id = task_entry.split("_")
                task_name = f"{task_name}_{search_space_id}"
                tasks.append(
                    HPOBTask(
                        task_name=task_name,
                        dataset_id=dataset_id,
                        root_dir=root_dir,
                        data_dir=root_dir / "data",
                    )
                )
            else:
                from src.tasks.mcts_transfer_task.func_task import BBOBTask

                task_name, func_seed = task_entry.split("_")
                tasks.append(
                    BBOBTask(
                        task_name=task_name,
                        data_dir=root_dir / "data",
                        func_seed=func_seed,
                    )
                )
        except:
            raise ValueError(f"Unknown task entry: {task_entry}")
    return tasks
