from typing import Union, List 
from pathlib import Path
import os 

def load_task_names(task_names: Union[str, List[str]], data_dir: Path) -> List[str]:
    if "," in task_names:
        task_names = list(task_names.split(","))
    elif task_names == "ALL":
        task_names = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath) and filename.endswith('.json'):
                if filename.startswith('HPOB') and not filename.startswith(('HPOB_5889', 'HPOB_7607', 'HPOB_5906')):
                    continue
                task_name = os.path.splitext(filename)[0]
                task_names.append(task_name)
    else:
        task_names = [task_names]
    return task_names