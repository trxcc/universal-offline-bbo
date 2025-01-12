import json
import os

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data2str.bboplace_bench_data import TASKNAMES as TASKNAMES_PLACE
from src.data2str.bboplace_bench_data import create_task as create_task_place
from src.data2str.co_data import TASKNAMES as TASKNAMES_CO
from src.data2str.co_data import create_task as create_task_co
from src.data2str.design_bench_data import TASKNAMES as TASKNAMES_DB
from src.data2str.design_bench_data import create_task as create_task_db
from src.data2str.mcts_transfer_data import (
    create_task_bbob,
    create_task_hpob,
    create_task_real_world,
)
from src.data2str.soo_bench_data import TASKNAMES as TASKNAMES_SB
from src.data2str.soo_bench_data import create_task as create_task_sb
from src.tasks.mcts_transfer_task.utils import load_mcts_transfer_data

data_dir = root / "data"
os.makedirs(data_dir, exist_ok=True)

# Generate Design-Bench data
for task_name in TASKNAMES_DB:
    print(task_name)
    task, metadata, data = create_task_db(task_name)

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": round(y.item(), 4)})

    output_file = f"{data_dir}/{task_name}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)

    metadata_file = f"{data_dir}/{task_name}.metadata"
    with open(metadata_file, "w") as f:
        f.write(metadata.to_string())

# Generate SOO-Bench data
for benchmark_id in [2, 3, 4, 6]:
    task_desc = f"gtopx_data_{benchmark_id}_1"
    assert task_desc in TASKNAMES_SB
    print(task_desc)

    task, metadata, data = create_task_sb(
        "gtopx_data", benchmark_id, 1, low=25, high=75
    )
    # such dataset settings follow Section 6 Para 1 in SOO-Bench paper
    # Link: https://openreview.net/pdf?id=bqf0aCF3Dd

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": round(y.item(), 4)})

    output_file = f"{data_dir}/{task_desc}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)

    metadata_file = f"{data_dir}/{task_desc}.metadata"
    with open(metadata_file, "w") as f:
        f.write(metadata.to_string())

# Generate BBOB data from MCTS-Transfer paper
data_dict = load_mcts_transfer_data(data_dir, "bbob")
for search_space_id, search_space_data in data_dict.items():
    # Read function seeds
    search_space_seeds = []
    for dataset_id in search_space_data.keys():
        search_space_seeds.append(eval(dataset_id.split("+")[2]))

    for seed in search_space_seeds:
        print(f"{search_space_id}_{seed}")
        task, metadata, data = create_task_bbob(search_space_id, data_dir, seed)
        task_desc = f"{search_space_id}_{seed}"

        task_data = []
        for x, y in zip(data.to_string(), task.y_np):
            task_data.append({"x": x, "y": round(y.item(), 4)})

        output_file = f"{data_dir}/{task_desc}.json"
        with open(output_file, "w") as f:
            json.dump(task_data, f, indent=2)

        metadata_file = f"{data_dir}/{task_desc}.metadata"
        with open(metadata_file, "w") as f:
            f.write(metadata.to_string())

# Generate real_world data from MCTS-Transfer paper
data_dict = load_mcts_transfer_data(data_dir, "real_world")
for search_space_id, search_space_data in data_dict.items():
    # Read function seeds
    search_space_seeds = []
    for dataset_id in search_space_data.keys():
        seed = eval(dataset_id.split("+")[2])
        if seed not in search_space_seeds:
            search_space_seeds.append(seed)

    for seed in search_space_seeds:
        print(f"{search_space_id}_{seed}")
        task, metadata, data = create_task_real_world(search_space_id, data_dir, seed)
        task_desc = f"{search_space_id}_{seed}"

        task_data = []
        for x, y in zip(data.to_string(), task.y_np):
            task_data.append({"x": x, "y": round(y.item(), 4)})

        output_file = f"{data_dir}/{task_desc}.json"
        with open(output_file, "w") as f:
            json.dump(task_data, f, indent=2)

        metadata_file = f"{data_dir}/{task_desc}.metadata"
        with open(metadata_file, "w") as f:
            f.write(metadata.to_string())

data_dict = load_mcts_transfer_data(data_dir, "hpob-data")
for search_space_id, search_space_data in data_dict.items():
    # Read function seeds
    dataset_ids = []
    for dataset_id in search_space_data.keys():
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)

    for dataset_id in dataset_ids:
        print(f"HPOB_{search_space_id}_{dataset_id}")
        task, metadata, data = create_task_hpob(
            search_space_id, dataset_id, root_dir=data_dir, data_dir=data_dir
        )
        task_desc = f"HPOB_{search_space_id}_{dataset_id}"

        task_data = []
        for x, y in zip(data.to_string(), task.y_np):
            task_data.append({"x": x, "y": round(y.item(), 4)})

        output_file = f"{data_dir}/{task_desc}.json"
        with open(output_file, "w") as f:
            json.dump(task_data, f, indent=2)

        metadata_file = f"{data_dir}/{task_desc}.metadata"
        with open(metadata_file, "w") as f:
            f.write(metadata.to_string())


for task_name in TASKNAMES_PLACE:
    print(task_name)
    task, metadata, data = create_task_place(task_name, root)

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": round(y.item(), 4)})

    output_file = f"{data_dir}/{task_name}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)

    metadata_file = f"{data_dir}/{task_name}.metadata"
    with open(metadata_file, "w") as f:
        f.write(metadata.to_string())

for task_name in TASKNAMES_CO:
    print(task_name)
    task, metadata, data = create_task_co(task_name, root / "data")

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": round(y.item(), 4)})

    output_file = f"{data_dir}/{task_name}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)

    metadata_file = f"{data_dir}/{task_name}.metadata"
    with open(metadata_file, "w") as f:
        f.write(metadata.to_string())
