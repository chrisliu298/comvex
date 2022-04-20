import numpy as np
import torch
from easydict import EasyDict


def get_weights(parameters):
    return torch.cat([param.flatten() for param in parameters]).to("cuda")


def get_stats(parameters):
    qs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to("cuda")
    stats = []
    for param in parameters:
        param = param.float()
        stats.append(
            torch.cat(
                [torch.mean(param).unsqueeze(0), torch.var(param).unsqueeze(0)]
            ).to("cuda")
        )
        stats.append(torch.quantile(param.flatten(), q=qs).to("cuda"))
    return torch.cat(stats).to("cuda")


def sample_hparams():
    hparams = EasyDict()
    initializations = ["xavier", "he", "orthogonal", "normal"]
    optimizers = ["adam", "sgd"]
    hparams.optimizer = optimizers[np.random.choice(len(optimizers))]
    hparams.lr = np.random.uniform(5e-4, 5e-2)
    hparams.weight_decay = np.random.uniform(1e-8, 1e-2)
    hparams.dropout_p = np.random.uniform(0, 0.5)
    hparams.initialization = initializations[np.random.choice(len(initializations))]
    return hparams
