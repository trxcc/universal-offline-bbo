from rootutils import rootutils

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.tasks import get_tasks

SOO_BENCH_TASKS = [
    "gtopx_data_2_1",
    "gtopx_data_3_1",
    "gtopx_data_4_1",
    "gtopx_data_6_1",
]
tasks = get_tasks(SOO_BENCH_TASKS, root_dir / "data")
print(tasks[0].y_np.max())
print(tasks[1].y_np.max())
print(tasks[2].y_np.max())
print(tasks[3].y_np.max())
