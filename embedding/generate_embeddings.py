import json
import os

import torch
from easydict import EasyDict
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import COMVEXConv, COMVEXLinear

HIDDEN_SIZES = [16, 16, 16, 10]
num_wokrers = int(os.cpu_count() / 2)
model_config_file = "cnnzoo1-cifar10gs-comvex-conv"
config = json.load(open(model_config_file + ".json"))
hparams = EasyDict(**config)

# Load model
model = (
    COMVEXLinear(**hparams) if "linear" in model_config_file else COMVEXConv(**hparams)
)
state_dict = torch.load(model_config_file + ".pt")
model.load_state_dict(state_dict)

# Load datasets
data = torch.load("val.pt")
data.keys()
if "linear" in model_config_file:
    dataset = TensorDataset(*[data[f"w{i + 1}"] for i in range(4)], data["test_acc"])
else:
    dataset = TensorDataset(
        *[
            data[f"w{i + 1}"].view(len(data["test_acc"]), 1, h, -1)
            for i, h in zip(range(4), HIDDEN_SIZES)
        ],
        data["test_acc"],
    )
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=num_wokrers)

# Generate embeddings
true_acc = []
pred_acc = []
embeddings = []

for batch in tqdm(dataloader):
    x1, x2, x3, x4, y = batch
    embedding, output = model([x1, x2, x3, x4])
    true_acc.append(y)
    pred_acc.append(output)
    embeddings.append(torch.hstack(embedding))

true_acc, pred_acc, embeddings = map(torch.cat, [true_acc, pred_acc, embeddings])

# Save results
results = {
    "true_acc": true_acc.detach(),
    "pred_acc": pred_acc,
    "embeddings": embeddings.detach(),
}
torch.save(results, model_config_file + "-embeddings.pt")
