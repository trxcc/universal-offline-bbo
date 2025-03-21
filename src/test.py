import os

import rootutils

# from transformers import AutoTokenizer

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from src.data.string_datamodule import StringXYDataModule

# data_module = StringXYDataModule(
#     "Superconductor-RandomForest-v0,AntMorphology-Exact-v0,DKittyMorphology-Exact-v0",
#     data_dir=root / "data",
#     # tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
#     val_ratio=0.2,
#     batch_size=1,
#     num_workers=16,
#     pin_memory=False,
# )

# data_module.setup(stage="fit")
# print(data_module.data_train)

# train_loader = data_module.train_dataloader()
# for batch_x, batch_y, batch_m in train_loader:
#     print(batch_x, batch_y, batch_m)
#     print(type(batch_x), type(batch_y), type(batch_m))
#     exit()

from src.searcher.cde import CDESearcher
from src.searcher.ga import GASearcher
from src.searcher.pso import PSOSearcher
from src.tasks.bboplace_bench_task import BBOPlacementTask
from src.tasks.co_task.kp import KPTask
from src.tasks.co_task.tsp import TSPTask
from src.tasks.mcts_transfer_task.func_task import BBOBTask, HPOBTask, RealWorldTask
from src.tasks.soo_bench_task import SOOBenchTask

# task = DesignBenchTask("Superconductor-RandomForest-v0")
# from src.tasks.soo_bench_task import SOOBenchTask

# from src.tasks.design_bench_task import DesignBenchTask

task = SOOBenchTask("gtopx_data", 2, 1, low=25, high=75, num_data=10000)
# task = BBOBTask(task_name="GriewankRosenbrock", func_seed=0, data_dir=root / "data")
# task = RealWorldTask("Rover", root / "data", 150, reevaluate=False)
# task = HPOBTask(
#     f"HPOB_5889", dataset_id="31", root_dir=root / "data", data_dir=root / "data"
# )
# task = BBOPlacementTask("adaptec1", root, reevaluate=True)
# task = TSPTask(problem_size=50, data_dir=root / "data")
ndim_problem = task.x_np.shape[1]
lb = task.x_np.min(axis=0)
ub = task.x_np.max(axis=0)
# searcher = CDESearcher(
#     problem={
#         "fitness_function": task.evaluate,
#         "ndim_problem": ndim_problem,
#         "lower_boundary": lb,
#         "upper_boundary": ub,
#     },
#     options={
#         "max_function_evaluations": 5000,
#         "n_individuals": 128,
#     },
#     task=task,
#     num_solutions=128,
#     MAXIMIZE=True,
# )
searcher = GASearcher(
    task=task,
    score_fn=lambda x: task._evaluate(x).reshape(-1, 1),
    num_solutions=128,
    n_gen=200,
)
x_res = searcher.run()
print(x_res.shape)
y_res = task.evaluate(x_res)
print(y_res)
