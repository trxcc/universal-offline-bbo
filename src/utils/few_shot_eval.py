from lightning import LightningModule
from pathlib import Path 
import numpy as np 
from typing import Tuple 
import json 
import os 
import torch
from tqdm import tqdm 
from copy import deepcopy
from transformers import T5Tokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from src.tasks import get_tasks
from src.searcher.ga import GASearcher
from src.data.components.omnipred_dataset import OmnipredDataset
from src.data.components.tokenizer import P10Tokenizer
from src.utils.data_transformation import omnipred_fitness_function_string
from src.utils.io_utils import save_metric_to_csv

few_shot_size = 100

class CustomFewShotGASearcher(GASearcher):
    def get_initial_designs(
    self, x: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        xl, xu = self.task.bounds
        
        # 获取y中最差的few_shot_size个解的索引
        worst_indices = np.argsort(y)[-few_shot_size:]
        worst_solutions = x[worst_indices]
        
        if k <= few_shot_size:
            # 从最差的few_shot_size个解中选择最好的k个
            best_k_indices = np.argsort(y[worst_indices])[:k]
            solutions = worst_solutions[best_k_indices]
        else:
            # 使用所有few_shot_size个最差解
            solutions = worst_solutions
            # 再随机生成额外的解
            additional_solutions = np.random.uniform(
                low=xl, high=xu, 
                size=(k - few_shot_size, len(xl))
            )
            # 合并两部分解
            solutions = np.vstack([solutions, additional_solutions])
        
        return solutions, None

def few_shot_eval(model: LightningModule, seed: int, run_name: str):
    task_names = ['Rover_150', 'LunarLander_100', 'RobotPush_100']
    tasks = get_tasks(task_names, Path("./"))
    score_dict = {} 
    root_dir = Path("./")
    for task_name, task_instance in zip(task_names, tasks):
        model0 = deepcopy(model)
        with open(f"./data/{task_name}.metadata", "r") as f:
            m = f.read()
        with open(f"./data/{task_name}.json", "r") as f:
            data = json.load(f)
        y = [d["y"] for d in data]
        x = [", ".join(d["x"]) for d in data]
        metadata = [m for _ in range(len(x))]
        task_names_list = [task_name for _ in range(len(x))]

        dataset = OmnipredDataset(
            x_data=x, y_data=y,
            poorest_data_size=few_shot_size,
            input_tokenizer=T5Tokenizer.from_pretrained('google-t5/t5-small', cache_dir='./'),
            output_tokenizer=P10Tokenizer(),
            max_length=324, 
            concat_metadata=True,
            cat_front=True,
            metadatas=metadata,
            task_names_list=task_names_list
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
            num_workers=64,
            pin_memory=False,
            shuffle=True
        )
        opt = torch.optim.SGD(model0.parameters(), lr=5e-4)
        model0 = model0.cuda() 
        for epoch in range(5):
            total_loss = 0 
            for batch in tqdm(dataloader):
                opt.zero_grad() 
                outputs = model0.forward(
                    input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    decoder_input_ids=batch["decoder_input_ids"].cuda(),
                    decoder_attention_mask=batch["decoder_attention_mask"].cuda(),
                    labels=batch["labels"].cuda(),
                )
                loss = outputs.loss
                loss.backward() 
                total_loss += loss.item() 
                opt.step() 
            print(f"Epoch {epoch}: {total_loss / len(dataloader)}")

        searcher = CustomFewShotGASearcher(
            n_gen=200,
            MAXIMIZE=True,
            EVAL_STABILITY=False,
            task=task_instance,
            score_fn=lambda x: omnipred_fitness_function_string(
                x, m=m, model=model0, task_name=task_name
            ),
            num_solutions=128
        )
        x_res = searcher.run()

        tmp_dict = task_instance.evaluate(x_res, return_normalized_y=True)
        res_dict = {}
        for k, v in tmp_dict.items():
            res_dict[f"{task_name}/{k}"] = v

        score_dict.update(res_dict)

        print("Final score statistics:")
        csv_dir = root_dir / "few_shot_csv_results"
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






