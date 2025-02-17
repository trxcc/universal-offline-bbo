import json
import os

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from torch.utils.data import DataLoader, Dataset, random_split, Subset
from src.utils.io_utils import load_task_names
from src.data.components.text_value_dataset import TextValueDataset
from transformers import T5Tokenizer, T5EncoderModel, T5Config
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


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
    # "Superconductor-RandomForest-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    # "HopperController-Exact-v0",
]
task_names = load_task_names(DESIGN_BENCH_TASKS,data_dir)
x_values = []
y_values = []
metadatas = []
task_names_list = []
for task_name in task_names:
    data_file = f"{data_dir}/{task_name}.json"
    assert os.path.exists(data_file)
    with open(data_file, "r") as f:
        data = json.load(f)

    ys = [d["y"] for d in data]
    xs = [", ".join(d["x"]) for d in data]
    y_values.extend(ys)
    x_values.extend(xs)
    assert len(xs) == len(ys)

    metadata_file = f"{data_dir}/{task_name}.metadata"
    with open(metadata_file, "r") as f:
        metadata = f.read()
        metadatas.extend([metadata for _ in range(len(xs))])
    task_names_list.extend([task_name for _ in range(len(xs))])

dataset = TextValueDataset(
    x_values,
    y_values,
    tokenizer=tokenizer,
    tokenizer_max_length=tokenizer_max_length,
    concat_metadata=True,
    metadatas=metadatas,
    task_names=task_names_list,
)
lengths = [
        len(x_values) - int(len(x_values) * val_ratio),
        int(len(x_values) * val_ratio),
]
train_dataset, _ = random_split(
    dataset=dataset, lengths=lengths
)

samples_per_task = 1000
all_embeddings = []
all_labels = []

for task_name in DESIGN_BENCH_TASKS:
    task_indices = [i for i in range(len(train_dataset)) 
                   if train_dataset.dataset.task_names[train_dataset.indices[i]] == task_name]
    
    # 随机采样100个样本
    sampled_indices = random.sample(task_indices, samples_per_task)
    task_samples = Subset(train_dataset, sampled_indices)
    
    # 获取embeddings
    dataloader = DataLoader(task_samples, batch_size=32)
    task_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch["text"]
            encoded_input = x.to('cuda')
            x_emb = embedder(**encoded_input)
            x_emb = mean_pooling(x_emb, encoded_input["attention_mask"].to('cuda'))
            embeddings = x_emb.cpu().numpy()
            task_embeddings.append(embeddings)

    
    task_embeddings = np.vstack(task_embeddings)
    all_embeddings.append(task_embeddings)
    all_labels.extend([task_name] * samples_per_task)

# 合并所有embeddings
all_embeddings = np.vstack(all_embeddings)

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# 绘制散点图
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b', 'y']
for i, task in enumerate(DESIGN_BENCH_TASKS):
    mask = [label == task for label in all_labels]
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
               c=colors[i], label=task, alpha=0.6)

plt.title('t-SNE visualization of task embeddings (train set)')
plt.legend()
plt.savefig('task_embeddings_tsne_train.png')
plt.close()