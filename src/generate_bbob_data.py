import json
import os
import numpy as np
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

data_dir = root / "data" / "mcts_transfer_data" / "bbob"
os.makedirs(data_dir, exist_ok=True)

from src.tasks.mcts_transfer_task.functions.colm_bbob import (
    Sphere,
    Rastrigin,
    BuecheRastrigin,
    LinearSlope,
    AttractiveSector,
    StepEllipsoidal,
    RosenbrockRotated,
    Ellipsoidal,
    Discus,
    BentCigar,
    SharpRidge,
    DifferentPowers,
    Weierstrass,
    SchaffersF7,
    SchaffersF7IllConditioned,
    GriewankRosenbrock,
    Schwefel,
    Katsuura,
    Lunacek,
    Gallagher101Me,
    Gallagher21Me,
    NegativeSphere,
    NegativeMinDifference,
    FonsecaFleming,
)

_name2func = {
    "Sphere": Sphere,
    "Rastrigin": Rastrigin,
    "BuecheRastrigin": BuecheRastrigin,
    "LinearSlope": LinearSlope,
    "AttractiveSector": AttractiveSector,
    "StepEllipsoidal": StepEllipsoidal,
    "RosenbrockRotated": RosenbrockRotated,
    "Ellipsoidal": Ellipsoidal,
    "Discus": Discus,
    "BentCigar": BentCigar,
    "SharpRidge": SharpRidge,
    "DifferentPowers": DifferentPowers,
    "Weierstrass": Weierstrass,
    "SchaffersF7": SchaffersF7,
    "SchaffersF7IllConditioned": SchaffersF7IllConditioned,
    "GriewankRosenbrock": GriewankRosenbrock,
    "Schwefel": Schwefel,
    "Katsuura": Katsuura,
    "Lunacek": Lunacek,
    "Gallagher101Me": Gallagher101Me,
    "Gallagher21Me": Gallagher21Me,
    "NegativeSphere": NegativeSphere,
    "NegativeMinDifference": NegativeMinDifference,
    "FonsecaFleming": FonsecaFleming,
}

def generate_samples(func_name, dim=10, num_samples=30, seed=240):
    # 设置随机种子
    np.random.seed(seed)
    
    # 初始化函数实例，此时会随机生成shift参数c
    func = _name2func[func_name](dim=dim)
    
    # 随机采样
    X = []
    Y = []
    for _ in range(num_samples):
        x = np.random.uniform(-5, 5, dim)
        try:
            y = float(func(x))
        except:
            print(func_name)
            print(func(x))
            print(func(x).shape)
            assert 0
        X.append(x.tolist())
        Y.append(y)
    
    return {
        f"Random+{seed}+{seed}": {
            "X": X,
            "Y": Y,
            "c": func.c.tolist()  # 记录shift参数
        }
    }

def generate_multiple_samples(func_names, dim=10, num_samples=30, num_shifts=5):
    # 创建包含函数名的顶层字典
    all_data = {}
    seed = 240
    for func_name in func_names.keys():
        all_data[func_name] = {}
    
        for i in range(num_shifts):
            seed += 1  # 使用不同的种子生成不同的shift
            data = generate_samples(func_name, dim, num_samples, seed)
            # 将每组数据添加到函数名下
            all_data[func_name].update(data)
    
        # 保存所有结果到一个文件
    output_file = data_dir / f"all_func_all_samples_{dim}d.json"
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    return all_data

# 生成样本
data = generate_multiple_samples(_name2func, dim=4, num_samples=30, num_shifts=50)