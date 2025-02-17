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
from scipy import stats

task_dict = {
    "AntMorphology-Exact-v0": "Ant",
    "DKittyMorphology-Exact-v0": "D'Kitty",
    "Superconductor-RandomForest-v0": "Superconductor",
    "TFBind8-Exact-v0": "TF Bind 8",
    "TFBind10-Exact-v0": "TF Bind 10"
}
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
ckpt_path = "/root/autodl-tmp/universal-offline-bbo/logs/omni_latent.ckpt"
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

def process_x_list(x_list):
    # 直接从列表中提取数值
    values = []
    for item in x_list:
        # 分割字符串并获取值部分
        value = float(item.split(': ')[1].strip("'"))
        values.append(value)
    
    return np.array(values)

# assert 0

data_dir = "data/"
val_ratio = 0.2
input_tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path='google-t5/t5-small')
output_tokenizer = P10Tokenizer()
tokenizer_max_length = 324

DESIGN_BENCH_TASKS = [
    # "AntMorphology-Exact-v0",
    # "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    # "TFBind8-Exact-v0",
    # "TFBind10-Exact-v0",
    # "gtopx_data_2_1",
    # "gtopx_data_3_1",
    # "gtopx_data_4_1",
    # "gtopx_data_6_1",
    # "HopperController-Exact-v0",
]
task_names = load_task_names(DESIGN_BENCH_TASKS,data_dir)
x_values = []
y_values = []
metadatas = []
true_x_values = []
task_names_list = []
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

samples_per_task = 10000
all_embeddings = []
all_labels = []



def calculate_pairs_and_correlation():
    all_embedding_diffs = []
    all_y_diffs = []
    
    for task_name in DESIGN_BENCH_TASKS:
        # 获取当前任务的索引
        task_indices = [i for i in range(len(train_dataset)) 
                       if train_dataset.dataset.task_names_list[train_dataset.indices[i]] == task_name]
        
        # 随机采样100对样本(200个点)
        sampled_indices = random.sample(task_indices, 1000)
        task_samples = Subset(train_dataset, sampled_indices)
        
        # 获取embeddings
        dataloader = DataLoader(task_samples, batch_size=32)
        task_embeddings = []
        task_y_values = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to('cuda')
                attention_mask = batch["attention_mask"].to('cuda')
                
                encoder_outputs = embedder.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                encoder_hidden_states = encoder_outputs.last_hidden_state
                masked_states = encoder_hidden_states * attention_mask.unsqueeze(-1)
                flat_states = masked_states.view(masked_states.size(0), -1)
                # mean_pooled = mean_pooling(encoder_hidden_states, attention_mask)
                embeddings = flat_states.cpu().numpy()
                # embeddings = mean_pooled.cpu().numpy()
                task_embeddings.append(embeddings)
                task_y_values.extend(batch["value"].numpy())
        
        # 合并所有batch的embeddings
        task_embeddings = np.vstack(task_embeddings)
        # print(task_embeddings.shape)
        # print(task_y_values[0])
        # assert 0
        
        # 计算500对样本之间的差值
        for i in range(0, 1000, 2):
            emb_diff = np.linalg.norm(task_embeddings[i] - task_embeddings[i+1])
            y_diff = abs(task_y_values[i] - task_y_values[i+1])
            
            all_embedding_diffs.append(emb_diff)
            all_y_diffs.append(y_diff)
    
    # 计算Spearman相关系数
    correlation, p_value = stats.spearmanr(all_embedding_diffs, all_y_diffs)
    
    return correlation, p_value, all_embedding_diffs, all_y_diffs

correlation, p_value, emb_diffs, y_diffs = calculate_pairs_and_correlation()
print(f"Spearman correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
