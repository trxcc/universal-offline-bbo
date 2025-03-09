import os
from pathlib import Path
from typing import List, Union

import pandas as pd


def load_task_names(task_names: Union[str, List[str]], data_dir: Path) -> List[str]:
    if isinstance(task_names, list):
        return task_names
    if "," in task_names:
        task_names = list(task_names.split(","))
    elif task_names == "ALL":
        task_names = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath) and filename.endswith(".json"):
                if filename.startswith("HPOB") and not filename.startswith(
                    ("HPOB_5889", "HPOB_5906")
                ):
                    # if not filename.startswith(("TSP", "KP")):
                    continue
                task_name = os.path.splitext(filename)[0]
                task_names.append(task_name)
    elif task_names == 'BBOB':
        task_names = []
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath) and filename.endswith(".json"):
                if not filename.startswith(
                    (
                        "GriewankRosenbrock",
                        "Lunacek",
                        "Rastrigin",
                        "RosenbrockRotated",
                        "SharpRidge",
                        "Sphere",
                        "BuecheRastrigin",
                        "LinearSlope",
                        "AttractiveSector",
                        "StepEllipsoidal",
                        "Ellipsoidal",
                        "Discus",
                        "BentCigar",
                        "DifferentPowers",
                        "Weierstrass",
                        "SchaffersF7",
                        "SchaffersF7IllConditioned",
                        "GriewankRosenbrock",
                        "Schwefel",
                        "Katsuura",
                        "Gallagher101Me",
                        "Gallagher21Me",
                        "NegativeSphere",
                        "NegativeMinDifference",
                        "FonsecaFleming",
                    )
                ):
                    continue
                    # if not filename.startswith(("TSP", "KP")):
                task_name = os.path.splitext(filename)[0]
                task_names.append(task_name)
    else:
        task_names = [task_names]
    return task_names


def check_if_evaluated(
    results_dir: Path,
    task_name: str,
    model_name: str,
    seed: int,
    metric_name: str
) -> bool:
    csv_path = results_dir / f"{seed}-{metric_name}.csv"
    if not os.path.exists(csv_path):
        return False
    
    existing_df = pd.read_csv(csv_path, header=0, index_col=0)
    
    if task_name not in existing_df.index:
        return False
    
    if model_name not in existing_df.columns:
        return False
    
    return not pd.isna(existing_df.loc[task_name, model_name])



def save_metric_to_csv(
    results_dir: Path,
    task_name: str,
    model_name: str,
    seed: int,
    metric_value: float,
    metric_name: str,
) -> None:
    csv_path = results_dir / f"{seed}-{metric_name}.csv"
    result = {"task": task_name, f"{model_name}": metric_value}

    if not os.path.exists(csv_path):
        new_df = pd.DataFrame([result])
        new_df.to_csv(csv_path, index=False)
    else:
        existing_df = pd.read_csv(csv_path, header=0, index_col=0)
        updated_df = existing_df.copy()
        updated_df.loc[task_name, f"{model_name}"] = metric_value
        updated_df.to_csv(csv_path, index=True, mode="w")
