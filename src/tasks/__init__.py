import os
from pathlib import Path
from typing import List, Optional, Tuple

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
    # "adaptec2",
    # "adaptec3",
    # "adaptec4",
    # "bigblue1",
    # "bigblue3",
]

REAL_WORLD_TASKS_PREFIX = ["LunarLander", "RobotPush", "Rover"]

CO_TASKS = [
    "KP_50",
    "KP_100",
    "KP_200",
    "TSP_20",
    "TSP_50",
    "TSP_100",
]


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
                    BBOPlacementTask(benchmark_name=task_entry, root_dir=root_dir)
                )
            elif task_entry in CO_TASKS:
                from src.tasks.co_task import KPTask, TSPTask

                task_name, problem_size = task_entry.split("_")
                if task_name == "KP":
                    tasks.append(
                        KPTask(
                            problem_size=int(problem_size), data_dir=root_dir / "data"
                        )
                    )
                else:
                    tasks.append(
                        TSPTask(
                            problem_size=int(problem_size), data_dir=root_dir / "data"
                        )
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


def get_tasks_from_suites(
    task_suites: str, root_dir: Path
) -> Tuple[List[str], List[OfflineBBOTask]]:
    assert task_suites.lower() in [
        "design_bench",
        "soo_bench",
        "bboplace_bench",
        "hpob",
        "bbob",
        "real_world",
        "co",
    ]
    if task_suites.lower() == "design_bench":
        task_names = DESIGN_BENCH_TASKS
    elif task_suites.lower() == "soo_bench":
        task_names = SOO_BENCH_TASKS
    elif task_suites.lower() == "bboplace_bench":
        task_names = BBOPLACE_BENCH_TASKS
    elif task_suites.lower() == "co":
        task_names = CO_TASKS
    elif task_suites.lower() == "hpob":
        task_names = []
        for filename in os.listdir(root_dir / "data"):
            filepath = os.path.join(root_dir / "data", filename)
            if os.path.isfile(filepath) and filename.endswith(".json"):
                if not filename.startswith(("HPOB_5889", "HPOB_5906")):
                    continue
                task_name = os.path.splitext(filename)[0]
                task_names.append(task_name)
    elif task_suites.lower() == "real_world":
        task_names = []
        for filename in os.listdir(root_dir / "data"):
            filepath = os.path.join(root_dir / "data", filename)
            if os.path.isfile(filepath) and filename.endswith(".json"):
                if not filename.startswith(("LunarLander", "RobotPush", "Rover")):
                    continue
                task_name = os.path.splitext(filename)[0]
                task_names.append(task_name)
    elif task_suites.lower() == "bbob":
        task_names = []
        for filename in os.listdir(root_dir / "data"):
            filepath = os.path.join(root_dir / "data", filename)
            if os.path.isfile(filepath) and filename.endswith(".json"):
                if not filename.startswith(
                    (
                        "GriewankRosenbrock",
                        "Lunacek",
                        "Rastrigin",
                        "RosenbrockRotated",
                        "SharpRidge",
                    )
                ):
                    continue
                task_name = os.path.splitext(filename)[0]
                task_names.append(task_name)
    return task_names, get_tasks(task_names, root_dir)
