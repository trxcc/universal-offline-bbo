import json
import os
from pathlib import Path
from typing import Dict

import numpy as np


def load_mcts_transfer_data(
    data_dir: Path,
    task_group: str,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    assert task_group in [
        "ant",
        "bbob",
        "dkitty",
        "hpob-data",
        "real_world",
        "Sphere2D",
        "superconductor",
    ]

    data_dir = data_dir / "mcts_transfer_data" / task_group
    assert os.path.exists(data_dir)

    if task_group == "hpob-data":
        data_file = data_dir / "meta-train-dataset.json"
    else:
        data_file = data_dir / "meta_dataset.json"

    with open(data_file, "r") as f:
        raw_data = json.load(f)

    return raw_data
