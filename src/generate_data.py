import json
import os

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data2str.design_bench_data import TASKNAMES as TASKNAMES_DB
from src.data2str.design_bench_data import create_task as create_task_db
from src.data2str.soo_bench_data import TASKNAMES as TASKNAMES_SB
from src.data2str.soo_bench_data import create_task as create_task_sb
from src.tasks.mcts_transfer_task.utils import load_mcts_transfer_data

data_path = root / "data"
os.makedirs(data_path, exist_ok=True)

for task_name in TASKNAMES_DB:
    print(task_name)
    task, metadata, data = create_task_db(task_name)

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": y.item()})

    output_file = f"{data_path}/{task_name}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)

    metadata_file = f"{data_path}/{task_name}.metadata"
    with open(metadata_file, "w") as f:
        f.write(metadata.to_string())

for benchmark_id in [2, 3, 4, 6]:
    task_desc = f"gtopx_data_{benchmark_id}_1"
    assert task_desc in TASKNAMES_SB
    print(task_desc)

    task, metadata, data = create_task_sb(
        "gtopx_data", benchmark_id, 1, low=25, high=75
    )

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": y.item()})

    output_file = f"{data_path}/{task_desc}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)

    metadata_file = f"{data_path}/{task_desc}.metadata"
    with open(metadata_file, "w") as f:
        f.write(metadata.to_string())

