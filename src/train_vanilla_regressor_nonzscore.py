import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import rootutils
import torch
import torch.nn as nn

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.xy_datamodule import XYDataModule
from src.models.components.mlp import SimpleMLP
from src.searcher.adam_batch import AdamSearcher
from src.searcher.ga import GASearcher
from src.tasks import get_tasks_from_suites
from src.tasks.base import OfflineBBOTask
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    model_fitness_function,
    task_wrapper,
)
from src.utils.io_utils import load_task_names, save_metric_to_csv, check_if_evaluated

log = RankedLogger(__name__, rank_zero_only=False)


def run_single_task(args: SimpleNamespace, task: OfflineBBOTask, task_name: str):
    print(task_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = task.x_np
    y = task.y_np
    # try:
    #     task._evaluate(task.x_np)
    # except:
    #     return

    datamodule = XYDataModule(task=task, num_workers=4, persistent_workers=False)
    datamodule.setup()

    model = SimpleMLP(
        input_dim=(
            task.ndim_problem * (task.num_classes - 1)
            if task.task_type == "Categorical"
            else task.ndim_problem
        ),
        hidden_dim=[2048, 2048],
        require_batch_norm=True,
    ).to(device)
    model = torch.compile(model)

    save_dir = root_dir / "expert_models"
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=0
    )
    criterion = nn.MSELoss(reduction="mean")

    if os.path.exists(save_dir / f"expert_seed_{args.seed}_task_{task_name}.pt"):
        model.load_state_dict(
            torch.load(save_dir / f"expert_seed_{args.seed}_task_{task_name}.pt")
        )

    else:
        for e in range(args.max_epochs):
            total_loss = 0
            for x_batch, y_batch in datamodule.train_dataloader():
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred.squeeze(), y_batch.squeeze())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss += loss.item()
            lr_scheduler.step()
            log.info(
                f"Epoch {e}, loss = {total_loss / len(datamodule.train_dataloader())}"
            )
            print(
                f"Epoch {e}, loss = {total_loss / len(datamodule.train_dataloader())}"
            )
        torch.save(
                model.state_dict(),
                save_dir / f"expert_seed_{args.seed}_task_{task_name}.pt"
            )
    csv_dir = root_dir / "post_csv_results"
    if not check_if_evaluated(
        results_dir=csv_dir,
        task_name=task_name,
        model_name="Expert_GA",
        seed=args.seed,
        metric_name="score-100th",
    ):
        searcher = GASearcher(
            n_gen=200,
            EVAL_STABILITY=task.eval_stability,
            score_fn=lambda x: model_fitness_function(x, model=model, task=task),
            task=task,
            num_solutions=128,
        )
        x_res = searcher.run()

        score_dict = {}
        tmp_dict = task.evaluate(x_res, return_normalized_y=True)
        res_dict = {}
        for k, v in tmp_dict.items():
            res_dict[f"{task_name}/{k}"] = v

        if task.eval_stability:
            X_all = searcher.X_all
            stability = task.evaluate_stability(X_all)
            res_dict[f"{task_name}/stability"] = stability

        score_dict.update(res_dict)

        log.info("Final score statistics:")
        csv_dir = root_dir / "few_shot_csv_results"
        for score_desc, score in res_dict.items():
            log.info(f"{score_desc}: {score}")
            print(f"{score_desc}: {score}")
            task_, metric_ = score_desc.split("/")
            save_metric_to_csv(
                results_dir=csv_dir,
                task_name=task_,
                model_name="Expert_GA",
                seed=args.seed,
                metric_value=score,
                metric_name=metric_,
            )

    if not check_if_evaluated(
        results_dir=csv_dir,
        task_name=task_name,
        model_name="Expert_Grad",
        seed=args.seed,
        metric_name="score-100th",
    ):
        searcher = AdamSearcher(
            n_steps=200,
            search_step_size=1e-3,
            model=model,
            EVAL_STABILITY=task.eval_stability,
            task=task,
            num_solutions=128,
            score_fn=lambda x: model(x),
        )

        x_res = searcher.run()

        score_dict = {}
        tmp_dict = task.evaluate(x_res, return_normalized_y=True)
        res_dict = {}
        for k, v in tmp_dict.items():
            res_dict[f"{task_name}/{k}"] = v

        if task.eval_stability:
            X_all = searcher.X_all
            stability = task.evaluate_stability(X_all)
            res_dict[f"{task_name}/stability"] = stability

        score_dict.update(res_dict)

        log.info("Final score statistics:")
        csv_dir = root_dir / "few_shot_csv_results"
        for score_desc, score in res_dict.items():
            log.info(f"{score_desc}: {score}")
            print(f"{score_desc}: {score}")
            task_, metric_ = score_desc.split("/")
            save_metric_to_csv(
                results_dir=csv_dir,
                task_name=task_,
                model_name="Expert_Grad",
                seed=args.seed,
                metric_value=score,
                metric_name=metric_,
            )


def run(args: SimpleNamespace):
    L.seed_everything(args.seed)
    task_names, tasks = get_tasks_from_suites(args.task_suites, root_dir)
    for task_name, task in zip(task_names, tasks):
        try:
            run_single_task(args, task, task_name)
        except AssertionError as e:
            print(f"Meet errors in running task: {task_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument(
    #     "--task_suites",
    #     type=str,
    #     default="design_bench",
    #     choices=[
    #         "design_bench",
    #         "soo_bench",
    #         "co",
    #         "real_world",
    #         "bboplace_bench",
    #         "bbob",
    #         "hpob",
    #     ],
    # )
    parser.add_argument(
        "--task",
        type=str, 
    )
    parser.add_argument("--max_epochs", type=int, default=200)
    args = parser.parse_args()
    from src.tasks import get_tasks
    task = get_tasks([args.task], Path("./"))[0]
    run_single_task(args, task, args.task)
