import json
import os

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from torch.utils.data import DataLoader, Dataset, random_split, Subset
from src.utils.io_utils import load_task_names
from src.data.components.plot_text_x_y_dataset import Text_x_ValueDataset
from transformers import T5Tokenizer, T5EncoderModel, T5Config
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


config = T5Config(
    d_model=384,
    d_ff=1536,
    num_layers=6,
    num_heads=6,
    vocab_size=32128,
    max_position_embeddings=256,
    decoder_start_token_id=None,
    use_cache=False
)
ckpt_path = "/root/autodl-tmp/universal-offline-bbo/logs/baseline_embed_regress_t5_m_cat_from_scratch_latent/runs/2025-01-20_12-22-13_seed42/checkpoints/epoch_epoch=149.ckpt"
embedder = T5EncoderModel(config)
checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda'))  # 如果用GPU可以改为'cuda'
model_state_dict = checkpoint['state_dict']

def mean_pooling(
        model_output, attention_mask
    ) -> torch.Tensor:
        token_embeddings: torch.Tensor = model_output[
            0
        ]  # Shape: [batch_size, sequence_length, hidden_size]
        input_mask_expanded: torch.Tensor = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )  # Shape: [batch_size, sequence_length, hidden_size]

        sum_embeddings: torch.Tensor = torch.sum(
            token_embeddings * input_mask_expanded, 1
        )  # Shape: [batch_size, hidden_size]
        sum_mask: torch.Tensor = torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )  # Shape: [batch_size, hidden_size]

        return sum_embeddings / sum_mask  # Shape: [batch_size, hidden_size]


def on_load_checkpoint(checkpoint: dict):
    keys_list = list(checkpoint["state_dict"].keys())
    for key in keys_list:
        if "embedder" in key and "metadata" not in key:
            if "orig_mod." in key:
                deal_key = key.replace("embedder._orig_mod.", "")
                checkpoint["state_dict"][deal_key] = checkpoint["state_dict"][key]
                del checkpoint["state_dict"][key]
                # print(deal_key)
        else:
            del checkpoint["state_dict"][key]
    return checkpoint
new_state_dict = on_load_checkpoint(checkpoint)



# 加载权重
embedder.load_state_dict(new_state_dict["state_dict"])
embedder.to('cuda')  # 明确移到cuda
embedder.eval()  # 设置为评估模式

data_dir = "data/"
val_ratio = 0.2
tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path='google-t5/t5-small')
tokenizer_max_length = 256

DESIGN_BENCH_TASKS = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    # "TFBind8-Exact-v0",
    # "TFBind10-Exact-v0",
    # "HopperController-Exact-v0",
]
task_names = load_task_names(DESIGN_BENCH_TASKS,data_dir)
x_values = []
true_x_values = []
y_values = []
metadatas = []
task_names_list = []
save_dir = 'lipschitz_plots'
os.makedirs(save_dir, exist_ok=True)

def process_x_list(x_list):
    # 直接从列表中提取数值
    values = []
    for item in x_list:
        # 分割字符串并获取值部分
        value = float(item.split(': ')[1].strip("'"))
        values.append(value)
    
    return np.array(values)

def save_single_plot(task_lipschitz_factors, task_name):
    fig = plot_lipschitz_histogram(task_lipschitz_factors, task_name)
    save_path = os.path.join(save_dir, f'{task_name}_lipschitz.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存
    print(f'Saved plot to {save_path}')

def plot_lipschitz_histogram(task_lipschitz_factors, task_name, bins=20):
    # 获取数据
    data = task_lipschitz_factors[task_name]
    emb_factors = data['emb']
    x_factors = data['x']
    
    # 计算合适的范围
    min_val = min(np.min(emb_factors), np.min(x_factors))
    max_val = max(np.max(emb_factors), np.max(x_factors))
    
    # 创建直方图
    plt.figure(figsize=(8, 6))
    
    # 绘制两个直方图，使用alpha来设置透明度，使两个直方图都可见
    plt.hist(emb_factors, bins=bins, range=(min_val, max_val), 
             alpha=0.7, label='T5-XXL Embedding', color='blue')
    plt.hist(x_factors, bins=bins, range=(min_val, max_val), 
             alpha=0.7, label='Raw Input', histtype='step', color='orange',linewidth=4)
    
    # 设置标题和标签
    plt.title(f'{task_name}')
    plt.xlabel('Normalized Lipschitz Factor')
    plt.ylabel('Count')
    
    # 添加图例
    plt.legend()
    
    # 调整布局
    plt.tight_layout()
    
    return plt

for task_name in task_names:
    data_file = f"{data_dir}/{task_name}.json"
    assert os.path.exists(data_file)
    with open(data_file, "r") as f:
        data = json.load(f)

    ys = [d["y"] for d in data]
    xs = [", ".join(d["x"]) for d in data]
    t_xs = [process_x_list(d["x"]) for d in data]
    y_values.extend(ys)
    x_values.extend(xs)
    true_x_values.extend(t_xs)
    assert len(xs) == len(ys)

    metadata_file = f"{data_dir}/{task_name}.metadata"
    with open(metadata_file, "r") as f:
        metadata = f.read()
        metadatas.extend([metadata for _ in range(len(xs))])
    task_names_list.extend([task_name for _ in range(len(xs))])

dataset = Text_x_ValueDataset(
    x_values,
    true_x_values,
    y_values,
    tokenizer=tokenizer,
    tokenizer_max_length=tokenizer_max_length,
    concat_metadata=True,
    metadatas=metadatas,
    task_names=task_names_list,
)
# orig_dataset = 
lengths = [
        len(x_values) - int(len(x_values) * val_ratio),
        int(len(x_values) * val_ratio),
]
train_dataset, _ = random_split(
    dataset=dataset, lengths=lengths
)

task_lipschitz_factors = {}

for task_name in DESIGN_BENCH_TASKS:
    task_indices = [i for i in range(len(train_dataset)) 
                   if train_dataset.dataset.task_names[train_dataset.indices[i]] == task_name]
    
    # 使用该任务的所有训练数据
    task_samples = Subset(train_dataset, task_indices)
    dataloader = DataLoader(task_samples, batch_size=128)
    task_embeddings = []
    task_x_values = []
    task_y_values = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["text"]
            encoded_input = x.to('cuda')
            x_emb = embedder(**encoded_input)
            x_emb = mean_pooling(x_emb, encoded_input["attention_mask"].to('cuda'))
            embeddings = x_emb.cpu().numpy()
            task_embeddings.append(embeddings)
            task_x_values.extend(batch["x_value"])
            task_y_values.extend(batch["value"])

    
    task_embeddings = np.vstack(task_embeddings)
    task_x_values = np.vstack(task_x_values)
    task_y_values = np.array(task_y_values)

    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(task_embeddings)
    normalized_x_value = scaler.fit_transform(task_x_values)

    embedding_lipschitz_factors = []
    x_lipschitz_factors = []

    emb_distances = cdist(normalized_embeddings, normalized_embeddings, metric='euclidean')
    x_distances = cdist(normalized_x_value, normalized_x_value, metric='euclidean')

    for i in range(len(normalized_embeddings)):
        # Set distance to self as infinity to exclude it from nearest neighbor search
        emb_distances[i, i] = np.inf
        x_distances[i, i] = np.inf
        emb_nearest_neighbor_idx = np.argmin(emb_distances[i])
        x_nearest_neighbor_idx = np.argmin(x_distances[i])
        
        # Compute embedding distance
        embedding_dist = emb_distances[i, emb_nearest_neighbor_idx]
        x_dist = x_distances[i, x_nearest_neighbor_idx]
        
        if embedding_dist > 0 and x_dist > 0:  # Avoid division by zero
            # Compute Y Lipschitz factor
            emb_y_dist = abs(task_y_values[i] - task_y_values[emb_nearest_neighbor_idx])
            emb_y_lipschitz_factor = emb_y_dist / embedding_dist
            embedding_lipschitz_factors.append(emb_y_lipschitz_factor)
            x_y_dist = abs(task_y_values[i] - task_y_values[x_nearest_neighbor_idx])
            x_y_lipschitz_factor = x_y_dist / x_dist
            x_lipschitz_factors.append(x_y_lipschitz_factor)
    emb_d = normalized_embeddings.shape[1]
    x_d = normalized_x_value.shape[1]
    emb_lipschitz_factors = np.array(embedding_lipschitz_factors) / np.sqrt(emb_d)
    x_lipschitz_factors = np.array(x_lipschitz_factors) / np.sqrt(x_d)
    task_lipschitz_factors[task_name] = {
        'x': x_lipschitz_factors,
        'emb': emb_lipschitz_factors
    }

    save_single_plot(task_lipschitz_factors,task_name)




