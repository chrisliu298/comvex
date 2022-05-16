import json
import logging
import os
import warnings

import numpy as np
import torch
import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from scipy.stats import loguniform
from torch.utils.data import DataLoader
from torchinfo import summary

from cmd_args import parse_args
from datasets import load_dataset
from models import COMVEXConv, COMVEXLinear, FCNet

os.environ["WANDB_SILENT"] = "True"
warnings.filterwarnings("ignore")


INITIALIZERS = ["xavier", "he", "orthogonal", "original"]
OPTIMIZERS = ["adam", "adamw", "adamax", "nadam", "radam"]


def setup(args):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def sample_cnnzoo1_hparams():
    # CNN Zoo 1
    hparams = EasyDict(
        n_layers=np.random.choice(np.arange(2, 8)).item(),
        hidden_size=np.random.choice(np.arange(256, 512)).item(),
        dropout_p=np.random.uniform(0.0, 0.2),
        weight_decay=loguniform.rvs(1e-8, 1e-3).item(),
        lr=loguniform.rvs(2e-5, 2e-3).item(),
        optimizer=OPTIMIZERS[np.random.choice(len(OPTIMIZERS))],
        batch_size=np.random.choice([64, 128, 256, 512]).item(),
        initializer=INITIALIZERS[np.random.choice(len(INITIALIZERS))],
    )
    return hparams


def sample_cnnzoo2_hparams():
    # CNN Zoo 2
    hparams = EasyDict(
        n_layers=np.random.choice(np.arange(2, 8)).item(),
        hidden_size=np.random.choice(np.arange(64, 512)).item(),
        dropout_p=np.random.uniform(0.0, 0.7),
        weight_decay=loguniform.rvs(1e-5, 1e-2).item(),
        lr=loguniform.rvs(2e-5, 2e-3).item(),
        optimizer=OPTIMIZERS[np.random.choice(len(OPTIMIZERS))],
        batch_size=np.random.choice([64, 128, 256, 512]).item(),
        initializer=INITIALIZERS[np.random.choice(len(INITIALIZERS))],
    )
    return hparams


def init_model(model, hparams, num_params, hidden_sizes, verbose):
    if model == "comvex-linear":
        model = COMVEXLinear(
            in_features=num_params,
            **hparams,
        )
        model_info = summary(
            model,
            input_data=[[torch.rand(1, shape) for shape in num_params]],
            verbose=verbose,
        )
    elif model == "comvex-conv":
        model = COMVEXConv(
            in_features=[int(p / h) for p, h in zip(num_params, hidden_sizes)],
            **hparams,
        )
        model_info = summary(
            model,
            input_data=[
                [
                    torch.rand(1, *shape)
                    for shape in [
                        [1, h, int(p / h)] for p, h in zip(num_params, hidden_sizes)
                    ]
                ]
            ],
            verbose=verbose,
        )
    elif model in ["fc", "fc-stats"]:
        in_features = 56 if model == "fc-stats" else sum(num_params)
        model = FCNet(
            in_features=in_features,
            **hparams,
        )
        model_info = summary(model, input_shape=(1, in_features), verbose=verbose)
    return model, model_info


def train(args):
    if "cnnzoo1" in args.project_name:
        HIDDEN_SIZES = [16, 16, 16, 10]
        NUM_PARAMS = [160, 2320, 2320, 170]
    elif "cnnzoo2" in args.project_name:
        HIDDEN_SIZES = [64, 64, 64, 10]
        NUM_PARAMS = [640, 36928, 36928, 650]
    # IN_FEATURES = [int(p / h) for p, h in zip(NUM_PARAMS, HIDDEN_SIZES)]
    if args.config:
        config = json.load(open(args.config))
        hparams = EasyDict(
            n_layers=config["n_layers"],
            hidden_size=config["hidden_size"],
            dropout_p=config["dropout_p"],
            weight_decay=config["weight_decay"],
            lr=config["lr"],
            optimizer=config["optimizer"],
            batch_size=config["batch_size"],
            initializer=config["initializer"],
        )
        print(json.dumps(dict(hparams), indent=4))
    else:
        hparams = (
            sample_cnnzoo1_hparams()
            if "cnnzoo1" in args.project_name
            else sample_cnnzoo2_hparams()
        )
    if args.embedding_dim:
        hparams.embedding_dim = args.embedding_dim

    config = {**dict(hparams), **vars(args)}
    if args.wandb:
        wandb.init(
            project=args.project_name,
            entity="chrisliu298",
            config=config,
        )
    train_dataloader = DataLoader(
        load_dataset(args.train_dataset_path, args.model, HIDDEN_SIZES),
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        load_dataset(args.val_dataset_path, args.model, HIDDEN_SIZES),
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model, model_info = init_model(
        args.model, hparams, NUM_PARAMS, HIDDEN_SIZES, args.verbose
    )
    if args.wandb:
        wandb.log({"total_params": model_info.total_params})

    model_checkpoint_callback = ModelCheckpoint(
        dirpath="model_ckpt/",
        filename="{epoch}_{avg_val_loss}",
        monitor="avg_val_loss",
        save_top_k=5,
        mode="min",
        every_n_epochs=1,
    )
    early_stopping_callback = EarlyStopping(
        monitor="avg_val_loss",
        patience=args.patience,
        mode="min",
    )
    trainer = Trainer(
        gpus=-1,
        logger=WandbLogger(
            save_dir="wandb/",
            project="generalization-prediction",
        )
        if args.wandb
        else TensorBoardLogger(save_dir="lightning_logs/"),
        callbacks=[
            early_stopping_callback,
            model_checkpoint_callback,
            TQDMProgressBar(refresh_rate=0),
        ],
        max_epochs=args.max_epochs,
        benchmark=True,
        enable_model_summary=False,
        check_val_every_n_epoch=1,
    )

    # Training
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.validate(
        ckpt_path=model_checkpoint_callback.best_model_path,
        dataloaders=val_dataloader,
        verbose=args.verbose,
    )
    trainer.test(
        ckpt_path=model_checkpoint_callback.best_model_path,
        dataloaders=DataLoader(
            load_dataset(args.test_dataset_path, args.model, HIDDEN_SIZES),
            batch_size=hparams.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        verbose=args.verbose,
    )
    if args.wandb:
        wandb.finish(quiet=True)

    if args.config:
        model.load_from_checkpoint(model_checkpoint_callback.best_model_path)
        torch.save(model.state_dict(), f"{args.project_name}.pt")


def main():
    args = parse_args()
    args.num_workers = int(os.cpu_count() / 2)
    setup(args)
    train(args)


if __name__ == "__main__":
    main()
