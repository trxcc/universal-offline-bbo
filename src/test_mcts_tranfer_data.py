import json
import os

import numpy as np
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

data_dir = root / "data" / "mcts_transfer_data"

assert os.path.exists(data_dir)

with open(data_dir / "real_world" / "meta_dataset.json", "r") as f:
    data = json.load(f)
    X_all = []
    y_all = []
    m_all = []
    for search_space_id, search_space_data in data.items():
        search_spaces = []
        data_size = 0
        length = 0
        print("Search space:", search_space_id)
        for dataset_id, dataset in search_space_data.items():
            if int(dataset_id.split('+')[-1]) not in search_spaces:
                search_spaces.append(int(dataset_id.split('+')[-1]))
                data_size += np.array(dataset["X"]).shape[0]
                length += 1
            print(np.array(dataset["X"]).shape)
        print(len(list(set(search_spaces))))
        print(data_size / length)
        print(data_size)
    assert 0
print(data.keys())
