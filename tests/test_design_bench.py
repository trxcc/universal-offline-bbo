# import design_bench as db
# import numpy as np

# for task_name in [
#     "AntMorphology-Exact-v0",
#     "DKittyMorphology-Exact-v0",
#     "Superconductor-RandomForest-v0",
#     "TFBind8-Exact-v0",
#     "TFBind10-Exact-v0",
# ]:
#     task = db.make(task_name)
#     x = task.x[:5]
#     y_true = task.y[:5]
#     y_pred = task.predict(x)

#     relative_error = np.abs((y_pred - y_true) / y_true)
#     mean_relative_error = np.mean(relative_error)

#     print(f"\nTask: {task_name}")
#     print(f"Predictions: {y_pred}")
#     print(f"Ground truth: {y_true}")
#     print(f"Mean relative error: {mean_relative_error:.4f}")

#     threshold = 0.1
#     if mean_relative_error > threshold:
#         print(f"Warning: Mean relative error exceeds {threshold*100}%")

import design_bench as db
import numpy as np
import torch

task_name = "AntMorphology-Exact-v0"


def normalize_score(y):
    dic2y = np.load("./src/tasks/dic2y.npy", allow_pickle=True).item()
    y_min, y_max = dic2y[task_name]
    return (y - y_min) / (y_max - y_min)


task = db.make(task_name, relabel=True)
task.map_normalize_x()

x = task.x.copy()
y = task.y.copy()

x_init = torch.Tensor(x[np.argsort(y.squeeze())[-128:]])

y_max = normalize_score(y.max())
y_pred_max = normalize_score(task.predict(x_init).max())
y_pred_np_max = normalize_score(task.predict(x_init.numpy()).max())

print(y_max, y_pred_max, y_pred_np_max)
