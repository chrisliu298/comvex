import json
import logging
import os

import cmd_args
import numpy as np
import torch
import wandb
from datasets import CIFAR10DataModule, MNISTDataModule
from easydict import EasyDict
from models import CNN
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

datamodules = {
    "cifar10": CIFAR10DataModule,
    "mnist": MNISTDataModule,
}


def sample_hparams():
    hparams = EasyDict()
    initializations = ["xavier", "he", "orthogonal", "normal"]
    optimizers = ["adam", "sgd"]
    hparams.optimizer = optimizers[np.random.choice(len(optimizers))]
    hparams.lr = np.random.uniform(5e-4, 5e-2)
    hparams.dropout_p = np.random.uniform(0, 0.5)
    hparams.initialization = initializations[np.random.choice(len(initializations))]
    return hparams


def setup(args):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    if args.wandb:
        wandb.init(
            project=args.project_name,
            entity="chrisliu298",
            config=vars(args),
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(args):
    datamodule = datamodules[args.dataset](
        batch_size=args.batch_size,
        num_workers=int(os.cpu_count() / 2),
    )
    datamodule.download_data()
    hparams = sample_hparams()
    print(json.dumps(dict(hparams), indent=4))
    model_spec = args.model_name.split("-")[-1]
    n_units = (
        [args.channels] + [int(x) for x in model_spec.split("x")] + [args.output_size]
    )
    model = CNN(
        n_units=n_units,
        save_path=args.model_ckpt_path,
        **hparams,
    )

    input_size = (
        (1, args.channels * args.hw * args.hw)
        if "mlp" in args.model_name
        else (1, args.channels, args.hw, args.hw)
    )
    summary_info = summary(
        model,
        input_size=input_size,
    )
    if args.wandb:
        wandb.log({"total_params": summary_info.total_params})

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=0),
    ]
    logger = (
        WandbLogger(
            save_dir="wandb/",
            project="cifar10",
        )
        if args.wandb
        else True
    )

    trainer = Trainer(
        gpus=-1,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, datamodule=datamodule)
    train_metrics, test_metrics = trainer.test(model, datamodule=datamodule, verbose=0)
    print(train_metrics)
    print(test_metrics)
    if args.wandb:
        wandb.finish(quiet=True)


def main():
    args = cmd_args.parse_args()
    setup(args)
    train(args)


if __name__ == "__main__":
    main()
