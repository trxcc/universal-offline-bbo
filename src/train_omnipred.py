import os
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.searcher.base import BaseSearcher
from src.tasks import get_tasks, get_tasks_from_suites
from src.tasks.base import OfflineBBOTask
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    omnipred_fitness_function_string,
    task_wrapper,
)
from src.utils.io_utils import load_task_names, save_metric_to_csv, check_if_evaluated

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating task <{cfg.task._target_}>")
    task: OfflineBBOTask = hydra.utils.instantiate(cfg.task)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        # resume_from_checkpoint=cfg.get("resume_path")
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "task": task,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # exit()

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        if not ckpt_path and (cfg.ckpt_path is not None and cfg.ckpt_path != ""):
            ckpt_path = cfg.ckpt_path

        log.info(f"ckpt_path is {ckpt_path}")

        if ckpt_path is not None:
            log.info(f"loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, weights_only=True)
            try:
                model.load_state_dict(checkpoint)
            except:
                new_state_dict = {}
                for k, v in checkpoint.items():
                    new_key = k.replace("_orig_mod.", "")
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict)
        # torch.save(obj=model.state_dict(),f="./logs/omni_nocat.pt")
        # exit()
        # from src.utils.zero_shot_eval import zero_shot_eval
        from src.utils.few_shot_eval import few_shot_eval
        few_shot_eval(model, cfg.get('seed'), run_name=cfg.task_name)
        exit()
        # task_names = load_task_names(cfg.task_names, data_dir=root_dir / "data")
        # tasks = get_tasks(task_names, root_dir=root_dir)
        task_names, tasks = get_tasks_from_suites(cfg.test_suites, root_dir)
        score_dict = {}

        csv_dir = root_dir / "new_csv_results"
        os.makedirs(csv_dir, exist_ok=True)
        for task_name, task_instance in zip(task_names, tasks):
            # if check_if_evaluated(
            #     results_dir=csv_dir,
            #     task_name=task_name,
            #     model_name=cfg.task_name,
            #     seed=cfg.get("seed"),
            #     metric_name="score-100th",
            # ):
            #     continue
            log.info(f"Instantiating searcher <{cfg.searcher._target_}>")
            with open(f"./data/{task_name}.metadata", "r") as f:
                m = f.read()
            searcher: BaseSearcher = hydra.utils.instantiate(
                cfg.searcher,
                task=task_instance,
                score_fn=lambda x: omnipred_fitness_function_string(
                    x, m=m, model=model, task_name=task_name
                ),
                EVAL_STABILITY=task.eval_stability,
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

            log.info("Final score statistics:")
            csv_dir = root_dir / "new_csv_results"
            for score_desc, score in res_dict.items():
                log.info(f"{score_desc}: {score}")
                print(score_desc)
                task_, metric_ = score_desc.split("/")
                save_metric_to_csv(
                    results_dir=csv_dir,
                    task_name=task_,
                    model_name=cfg.task_name,
                    seed=cfg.get("seed"),
                    metric_value=score,
                    metric_name=metric_,
                )
                for logger0 in logger:
                    logger0.log_metrics({score_desc: score}, step=1)

        # trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        # log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics
    test_metrics = score_dict

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

# universal-offline-bbo/logs/omnipred_test/runs/2025-01-03_20-06-41_seed42/Universal/ednc6tch/checkpoints/epoch=199-step=21400.ckpt
# logs/omnipred_test/runs/2025-01-03_22-07-17_seed42/Universal/rn1x1r91/checkpoints/test.ckpt
# logs/omnipred_test/runs/2025-01-04_22-59-00_seed42/Universal/ur6l0g7m/checkpoints/test.ckpt
# logs/omnipred_test/runs/2025-01-05_17-11-07_seed42/Universal/2064go0t/checkpoints/test.ckpt
# logs/omnipred_24m/runs/2025-01-10_01-29-12_seed42/checkpoints/last.ckpt
# logs/baseline_omnipred_24m/runs/2025-01-13_17-27-20_seed42/checkpoints/last.ckpt
# logs/baseline_omnipred_24m/runs/2025-01-13_22-48-34_seed42/checkpoints/last.ckpt
# logs/baseline_omnipred_24m/runs/2025-01-13_23-39-10_seed42/checkpoints/last.ckpt
# logs/baseline_omnipred_24m/runs/2025-01-14_01-42-14_seed42/checkpoints/last.ckpt
