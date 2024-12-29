import json
import os

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data2str.design_bench_data import TASKNAMES, create_task



data_path = root / "data"
os.makedirs(data_path, exist_ok=True)

for task_name in TASKNAMES:
    print(task_name)
    task, metadata, data = create_task(task_name)

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": y.item()})

    output_file = f"{data_path}/{task_name}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)
    
    metadata_file = f"{data_path}/{task_name}.metadata"
    with open(metadata_file, "a") as f:
        f.write(metadata.to_string())
