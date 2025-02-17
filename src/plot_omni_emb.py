import json
import os

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from torch.utils.data import DataLoader, Dataset, random_split, Subset
from src.utils.io_utils import load_task_names
from src.data.components.text_value_dataset import TextValueDataset
from src.data.components.omnipred_dataset import OmnipredDataset
from transformers import T5Tokenizer, T5EncoderModel, T5Config, T5ForConditionalGeneration
from src.data.components.tokenizer import P10Tokenizer
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import matplotlib

params = {
    'lines.linewidth': 1.5,
    'legend.fontsize': 17,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
}
matplotlib.rcParams.update(params)


config = T5Config(
    vocab_size=32128,
    num_layers=6,
    num_decoder_layers=6,
    d_kv=32,
    d_model=384,
    d_ff=512,
    decoder_start_token_id=None,
    use_cache=False
)
ckpt_path = "/root/autodl-tmp/universal-offline-bbo/logs/omni.ckpt"
embedder = T5ForConditionalGeneration(config=config)
checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda'))  # 如果用GPU可以改为'cuda'
model_state_dict = checkpoint['state_dict']

def mean_pooling(
        model_output, attention_mask
    ) -> torch.Tensor:
        token_embeddings: torch.Tensor = model_output # Shape: [batch_size, sequence_length, hidden_size]
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
        # print(key)
        if "model" in key and "metadata" not in key:
            if "orig_mod." in key:
                deal_key = key.replace("model._orig_mod.", "")
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

# assert 0

data_dir = "data/"
val_ratio = 0.2
input_tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path='google-t5/t5-small')
output_tokenizer = P10Tokenizer()
tokenizer_max_length = 324

DESIGN_BENCH_TASKS = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    # "gtopx_data_2_1",
    # "gtopx_data_3_1",
    # "gtopx_data_4_1",
    # "gtopx_data_6_1",
    # "HopperController-Exact-v0",
]

task_dict = {
    "AntMorphology-Exact-v0": "Ant",
    "DKittyMorphology-Exact-v0": "D'Kitty",
    "Superconductor-RandomForest-v0": "Superconductor",
    "TFBind8-Exact-v0": "TF Bind 8",
    "TFBind10-Exact-v0": "TF Bind 10"
}

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
    # print(xs)
    # assert 0
    y_values.extend(ys)
    x_values.extend(xs)
    assert len(xs) == len(ys)

    metadata_file = f"{data_dir}/{task_name}.metadata"
    with open(metadata_file, "r") as f:
        metadata = f.read()
        metadatas.extend([metadata for _ in range(len(xs))])
    task_names_list.extend([task_name for _ in range(len(xs))])

dataset = OmnipredDataset(
    x_values,
    y_values,
    input_tokenizer=input_tokenizer,
    output_tokenizer=output_tokenizer,
    max_length=tokenizer_max_length,
    concat_metadata=True,
    metadatas=metadatas,
    task_names_list=task_names_list,
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
                   if train_dataset.dataset.task_names_list[train_dataset.indices[i]] == task_name]
    
    # 随机采样100个样本
    sampled_indices = random.sample(task_indices, samples_per_task)
    task_samples = Subset(train_dataset, sampled_indices)
    
    # 获取embeddings
    dataloader = DataLoader(task_samples, batch_size=32)
    task_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            # x = batch["text"]
            # encoded_input = x.to('cuda')
            # x_emb = embedder.encoder(**encoded_input)
            # x_emb = mean_pooling(x_emb, encoded_input["attention_mask"].to('cuda'))
            # embeddings = x_emb.cpu().numpy()
            # task_embeddings.append(embeddings)
            input_ids=batch["input_ids"].to('cuda')
            attention_mask=batch["attention_mask"].to('cuda')
            encoder_outputs = embedder.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
            encoder_hidden_states = encoder_outputs.last_hidden_state
            # print(encoder_hidden_states.size())
            # print(attention_mask.size())
            # assert 0
            masked_states = encoder_hidden_states * attention_mask.unsqueeze(-1)
            flat_states = masked_states.view(masked_states.size(0), -1)
            embeddings = flat_states.cpu().numpy()
            # mean_pooled = mean_pooling(encoder_hidden_states,attention_mask)
            # # proj_pool = embedder.projection_head(mean_pooled)
            # embeddings = mean_pooled.cpu().numpy()
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
ax = plt.subplot(111)
ax.set_position([0.125, 0.18, 0.8, 0.75])
colors = ['r', 'g', 'b', 'y', 'black']
for i, task in enumerate(DESIGN_BENCH_TASKS):
    mask = [label == task for label in all_labels]
    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
               c=colors[i], label=task_dict[task], alpha=0.6)

ax.set_title('t-SNE plot of Vanilla UniSO-T embeddings',fontsize=25)
ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.13), 
          ncol=5, frameon=True, handletextpad=0.5, columnspacing=1)
ax.set_xlabel('t-SNE dimension 1', fontsize=25)
ax.set_ylabel('t-SNE dimension 2', fontsize=25)
plt.rc('font', family='Arial')
plt.savefig('task_embeddings_tsne_train_omni_test.png')
plt.savefig('task_embeddings_tsne_train_omni_test.pdf')
plt.close()