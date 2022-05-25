import json
import os

import torch
from easydict import EasyDict
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import COMVEXConv, COMVEXLinear

dataset_names = ["cifar10gs", "mnist", "fashionmnist", "usps"]
HIDDEN_SIZES = [64, 64, 64, 10]
num_wokrers = int(os.cpu_count() / 2)

for dataset_name in dataset_names:
    os.system("rm *.pt* *.json*")
    os.system(
        f"rsync /content/drive/Shareddrives/COMVEX/cnnzoo2/predictors/*{dataset_name}* ."
    )
    os.system(
        f"rsync /content/drive/Shareddrives/COMVEX/cnnzoo2/{dataset_name}/train.pt ."
    )
    for model_type in ["linear", "conv"]:

        model_config_file = f"cnnzoo2-{dataset_name}-comvex-{model_type}"
        config = json.load(open(model_config_file + ".json"))
        hparams = EasyDict(**config)

        # Load model
        model = (
            COMVEXLinear(**hparams)
            if "linear" in model_config_file
            else COMVEXConv(**hparams)
        )
        state_dict = torch.load(model_config_file + ".pt")
        model.load_state_dict(state_dict)

        # Load datasets
        data = torch.load("train.pt")
        data.keys()
        if "linear" in model_config_file:
            dataset = TensorDataset(
                *[data[f"w{i + 1}"] for i in range(4)], data["test_acc"]
            )
        else:
            dataset = TensorDataset(
                *[
                    data[f"w{i + 1}"].view(len(data["test_acc"]), 1, h, -1)
                    for i, h in zip(range(4), HIDDEN_SIZES)
                ],
                data["test_acc"],
            )
        del data
        dataloader = DataLoader(
            dataset, batch_size=512, shuffle=False, num_workers=num_wokrers
        )

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

        true_acc, pred_acc, embeddings = map(
            torch.cat, [true_acc, pred_acc, embeddings]
        )

        # Save results
        results = {
            "true_acc": true_acc.detach_(),
            "pred_acc": pred_acc.detach_(),
            "embeddings": embeddings.detach_(),
        }
        path = "/content/drive/Shareddrives/COMVEX/cnnzoo2/embeddings/"
        torch.save(
            results, path + f"cnnzoo2_{dataset_name}_comvex-{model_type}_embeddings.pt"
        )
        del dataset
        del dataloader
    os.system("rm *.pt* *.json*")
