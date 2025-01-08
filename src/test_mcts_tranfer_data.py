import json
import os

import numpy as np
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

data_dir = root / "data" / "mcts_transfer_data"

assert os.path.exists(data_dir)

with open(data_dir / "bbob" / "meta_dataset.json", "r") as f:
    data = json.load(f)
    X_all = []
    y_all = []
    m_all = []
    for search_space_id, search_space_data in data.items():
        for dataset_id, dataset in search_space_data.items():
            print(search_space_id, dataset_id)
            print(np.array(dataset["X"]).shape)
    assert 0
print(data.keys())
