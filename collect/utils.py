import numpy as np
from easydict import EasyDict


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
