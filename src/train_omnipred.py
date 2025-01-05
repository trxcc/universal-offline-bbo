import os
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
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
from src.tasks.base import OfflineBBOTask
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    model_fitness_function_string,
    task_wrapper,
)

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

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.model.generation_config.decoder_start_token_id = (
        datamodule.output_tokenizer.bos_token_id
    )
    model.model.generation_config.bos_token_id = (
        datamodule.output_tokenizer.bos_token_id
    )
    model.model.generation_config.pad_token_id = (
        datamodule.output_tokenizer.pad_token_id
    )
    model.model.generation_config.eos_token_id = (
        datamodule.output_tokenizer.eos_token_id
    )

    log.info(f"Instantiating task <{cfg.task._target_}>")
    task: OfflineBBOTask = hydra.utils.instantiate(cfg.task)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
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
            checkpoint = torch.load(ckpt_path)
            try:
                model.load_state_dict(checkpoint["state_dict"])
            except:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_key = k.replace("_orig_mod.", "")
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict)

        datamodule.setup()
        model = model.cuda()
        for batch in datamodule.train_dataloader():
            for v in batch.values():
                v = v.cuda()
            preds = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            datamodule.output_tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
            print(
                batch,
                preds,
            )
            target_numbers = []
            for label in batch['labels']:
                try:
                    num = float(datamodule.output_tokenizer.decode(
                        label[label != -100], skip_special_tokens=True
                    ))
                    target_numbers.append(num)
                except ValueError:
                    target_numbers.append(float("-inf"))
            target_numbers = torch.tensor(target_numbers, device="cuda")
            
            pred_numbers = [] 
            for pred in preds:
                try:
                    num = float(datamodule.output_tokenizer.decode(
                        pred, skip_special_tokens=True
                    ))
                    pred_numbers.append(num)
                except ValueError:
                    pred_numbers.append(float("-inf"))
            pred_numbers = torch.tensor(pred_numbers, device='cuda')
            for t, p in zip(target_numbers.flatten(), pred_numbers.flatten()):
                print(t.item(), p.item())
            # assert 0, (target_numbers, pred_numbers)
            exit()

        log.info(f"Instantiating searcher <{cfg.searcher._target_}>")
        with open(f"./data/{cfg.task.task_name}.metadata", "r") as f:
            m = f.read()
        searcher: BaseSearcher = hydra.utils.instantiate(
            cfg.searcher,
            task=task,
            score_fn=lambda x: model_fitness_function_string(x, m=m, model=model),
            # universal-offline-bbo/logs/embed_regress_multitask_m_emb/runs/2024-12-30_00-28-09_seed42/checkpoints/last.ckpt
        )

        x_res = searcher.run()
        score_dict = task.evaluate(x_res, return_normalized_y=True)
        log.info("Final score statistics:")
        for score_desc, score in score_dict.items():
            log.info(f"{score_desc}: {score}")
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