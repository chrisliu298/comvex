import contextlib
import json
import logging
import os
import warnings

import numpy as np
import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from scipy.stats import loguniform
from torchinfo import summary

import cmd_args
from datasets import (
    CIFAR10DataModule,
    CIFAR10GSDataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
    SVHNDataModule,
    USPSDataModule,
)
from models import CNN

datamodules = {
    "cifar10": CIFAR10DataModule,
    "cifar10gs": CIFAR10GSDataModule,
    "mnist": MNISTDataModule,
    "fashionmnist": FashionMNISTDataModule,
    "svhn": SVHNDataModule,
    "usps": USPSDataModule,
}
os.environ["WANDB_SILENT"] = "True"
warnings.filterwarnings("ignore")


def sample_hparams():
    initializers = ["xavier", "he", "orthogonal"]
    optimizers = ["adam", "sgd"]
    activations = ["relu", "tanh"]

    hparams = EasyDict(
        optimizer=optimizers[np.random.choice(len(optimizers))],
        lr=loguniform.rvs(1e-4, 1e-2).item(),
        weight_decay=loguniform.rvs(1e-8, 1e-2).item(),
        dropout_p=np.random.uniform(0, 0.7),
        initializer=initializers[np.random.choice(len(initializers))],
        activation=activations[np.random.choice(len(activations))],
        training_frac=np.random.choice([0.1, 0.25, 0.5, 1.0]),
    )
    return hparams


def setup(args):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    if args.wandb:
        wandb.init(
            project=args.project_name,
            entity="chrisliu298",
            config=vars(args),
        )
    seed_everything(args.seed)


def train(args):
    hparams = sample_hparams()
    # if args.dataset == "svhn":
    #     hparams.optimizer = "adam"
    if args.verbose:
        print(json.dumps(dict(hparams), indent=4))
    datamodule = datamodules[args.dataset](
        batch_size=args.batch_size,
        num_workers=int(os.cpu_count() / 2),
    )
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            datamodule.download_data()
    datamodule.sample_dataset(size=hparams.training_frac)

    model_spec = args.model_name.split("-")[-1]
    n_units = (
        [args.channels] + [int(x) for x in model_spec.split("x")] + [args.output_size]
    )
    model = CNN(
        n_units=n_units,
        save_path=args.model_ckpt_path,
        seed=args.seed,
        **hparams,
    )
    summary_info = summary(
        model,
        input_size=(1, args.channels, args.hw, args.hw),
        verbose=args.verbose,
    )
    if args.wandb:
        wandb.log({"total_params": summary_info.total_params})

    logger = (
        WandbLogger(
            save_dir="wandb/",
            project=args.dataset,
        )
        if args.wandb
        else True
    )

    trainer = Trainer(
        gpus=-1,
        callbacks=[TQDMProgressBar(refresh_rate=0)],
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )
    trainer.fit(model=model, datamodule=datamodule)
    if args.wandb:
        wandb.finish(quiet=True)


def main():
    args = cmd_args.parse_args()
    setup(args)
    train(args)


if __name__ == "__main__":
    main()
