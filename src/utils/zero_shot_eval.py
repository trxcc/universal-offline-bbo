from lightning import LightningModule
from pathlib import Path 
from typing import Tuple
import numpy as np 
import os 

from src.tasks import get_tasks
from src.searcher.ga import GASearcher
from src.utils.data_transformation import omnipred_fitness_function_string
from src.utils.io_utils import save_metric_to_csv

class CustomZeroShotGASearcher(GASearcher):
    def get_initial_designs(
        self, x: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        xl, xu = self.task.bounds
        solutions = np.random.uniform(low=xl, high=xu, size=(k, len(xl)))
        return solutions, None

def zero_shot_eval(model: LightningModule, seed: int, run_name: str):
    task_names = ['RobotPush_100', 'Rover_150', 'LunarLander_100']
    tasks = get_tasks(task_names, Path("./"))
    score_dict = {} 
    root_dir = Path("./")
    for task_name, task_instance in zip(task_names, tasks):
        with open(f"./data/{task_name}.metadata", "r") as f:
            m = f.read()
        searcher = CustomZeroShotGASearcher(
            n_gen=200,
            MAXIMIZE=True,
            EVAL_STABILITY=False,
            task=task_instance,
            score_fn=lambda x: omnipred_fitness_function_string(
                x, m=m, model=model, task_name=task_name
            ),
            num_solutions=128
        )
        x_res = searcher.run()

        tmp_dict = task_instance.evaluate(x_res, return_normalized_y=True)
        res_dict = {}
        for k, v in tmp_dict.items():
            res_dict[f"{task_name}/{k}"] = v

        if task_instance.eval_stability:
            X_all = searcher.X_all
            stability = task_instance.evaluate_stability(X_all)
            res_dict[f"{task_name}/stability"] = stability

        score_dict.update(res_dict)

        print("Final score statistics:")
        csv_dir = root_dir / "zero_shot_csv_results"
        os.makedirs(csv_dir, exist_ok=True)
        for score_desc, score in res_dict.items():
            print(f"{score_desc}: {score}")
            print(score_desc)
            task_, metric_ = score_desc.split("/")
            save_metric_to_csv(
                results_dir=csv_dir,
                task_name=task_,
                model_name=run_name,
                seed=seed,
                metric_value=score,
                metric_name=metric_,
            )

        