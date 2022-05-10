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
from datasets import load_datasets
from models import COMVEXConv, COMVEXLinear, FCNet

os.environ["WANDB_SILENT"] = "True"
warnings.filterwarnings("ignore")

HIDDEN_SIZES = [64, 64, 64, 10]
NUM_PARAMS = [640, 36928, 36928, 650]
IN_FEATURES = [int(p / h) for p, h in zip(NUM_PARAMS, HIDDEN_SIZES)]

# HIDDEN_SIZES = [16, 16, 16, 10]
# NUM_PARAMS = [160, 2320, 2320, 170]
# IN_FEATURES = [int(p / h) for p, h in zip(NUM_PARAMS, HIDDEN_SIZES)]


def setup(args):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    # np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, linewidth=60, sci_mode=False)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # seed_everything(args.seed)


def sample_hparams():
    initializers = ["xavier", "he", "orthogonal", "uniform"]
    # optimizers = ["adam", "sgd"]
    hparams = EasyDict(
        n_layers=np.random.choice(np.arange(1, 5)).item(),
        # n_layers=1,
        hidden_size=np.random.choice(np.arange(256, 513)).item(),
        dropout_p=np.random.uniform(0, 0.5),
        weight_decay=loguniform.rvs(1e-8, 1e-3).item(),
        lr=loguniform.rvs(1e-4, 1e-2).item(),
        # optimizer=optimizers[np.random.choice(len(optimizers))],
        optimizer="adam",
        batch_size=np.random.choice([64, 128, 256, 512]).item(),
        initializer=initializers[np.random.choice(len(initializers))],
    )
    return hparams


def init_model(model, hparams, verbose):
    if model == "comvex-linear":
        model = COMVEXLinear(
            in_features=NUM_PARAMS,
            **hparams,
        )
        model_info = summary(
            model,
            input_data=[[torch.rand(1, shape) for shape in NUM_PARAMS]],
            verbose=verbose,
        )
    elif model == "comvex-conv":
        model = COMVEXConv(
            in_features=[int(p / h) for p, h in zip(NUM_PARAMS, HIDDEN_SIZES)],
            **hparams,
        )
        model_info = summary(
            model,
            input_data=[
                [
                    torch.rand(1, *shape)
                    for shape in [
                        [1, h, int(p / h)] for p, h in zip(NUM_PARAMS, HIDDEN_SIZES)
                    ]
                ]
            ],
            verbose=verbose,
        )
    elif model in ["fc", "fc-stats"]:
        in_features = 56 if model == "fc-stats" else sum(NUM_PARAMS)
        model = FCNet(
            in_features=in_features,
            **hparams,
        )
        model_info = summary(model, input_shape=(1, in_features), verbose=verbose)
    return model, model_info


def train(args):
    hparams = sample_hparams()
    if args.embedding_dim:
        hparams.embedding_dim = args.embedding_dim

    print(json.dumps(dict(hparams), indent=4))
    config = {**dict(hparams), **vars(args)}
    if args.wandb:
        wandb.init(
            project=args.project_name,
            entity="chrisliu298",
            config=config,
        )

    train_dataset, val_dataset, test_dataset = load_datasets(
        args.train_data_path,
        args.val_data_path,
        args.test_data_path,
        args.model,
        HIDDEN_SIZES,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model, model_info = init_model(args.model, hparams, args.verbose)
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
        dataloaders=test_dataloader,
        verbose=args.verbose,
    )
    if args.wandb:
        wandb.finish(quiet=True)


def main():
    args = parse_args()
    args.num_workers = int(os.cpu_count() / 2)
    setup(args)
    train(args)


if __name__ == "__main__":
    main()
