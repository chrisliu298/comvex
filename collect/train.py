import json
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
from scipy.stats import loguniform
from torchinfo import summary

datamodules = {
    "cifar10": CIFAR10DataModule,
    "mnist": MNISTDataModule,
}


def sample_hparams():
    initializations = ["xavier", "he", "orthogonal", "normal"]
    optimizers = ["adam", "sgd"]
    activations = ["relu", "tanh"]

    hparams = EasyDict(
        # optimizer=optimizers[np.random.choice(len(optimizers))],
        optimizer="sgd",
        # lr=loguniform.rvs(5e-4, 5e-2).item(),
        lr=1e-2,
        # weight_decay=loguniform.rvs(1e-8, 1e-2).item(),
        weight_decay=0,
        # dropout_p=np.random.uniform(0, 0.5),
        dropout_p=0,
        # initialization=initializations[np.random.choice(len(initializations))],
        # initialization="xavier",
        # activation=activations[np.random.choice(len(activations))],
        activation="relu",
        # training_frac=np.random.choice([0.1, 0.25, 0.5, 1.0]),
        training_frac=1.0,
    )
    return hparams


def setup(args):
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    if args.wandb:
        wandb.init(
            project=args.project_name,
            entity="chrisliu298",
            config=vars(args),
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(args):
    setup(args)
    hparams = sample_hparams()
    # print(json.dumps(dict(hparams), indent=4))
    datamodule = datamodules[args.dataset](
        batch_size=args.batch_size,
        num_workers=int(os.cpu_count() / 2),
    )
    datamodule.download_data()
    datamodule.sample_dataset(size=hparams.training_frac)

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
        verbose=args.verbose,
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
        enable_model_summary=False,
    )
    trainer.fit(model=model, datamodule=datamodule)
    train_metrics, test_metrics = trainer.test(model, datamodule=datamodule, verbose=0)
    print(train_metrics)
    print(test_metrics)
    if args.wandb:
        wandb.finish(quiet=True)


def main():
    args = cmd_args.parse_args()
    train(args)


if __name__ == "__main__":
    main()
