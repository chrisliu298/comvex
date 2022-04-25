import json
import logging
import os

import numpy as np
import torch
import wandb
from cmd_args import parse_args
from datasets import load_datasets
from easydict import EasyDict
from models import COMVEXConv, COMVEXLinear, FullyConnected
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from scipy.stats import loguniform
from torch.utils.data import DataLoader
from torchinfo import summary


def setup(args):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, linewidth=60, sci_mode=False)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)


def sample_hparams():
    initializations = ["xavier", "he", "orthogonal"]
    # optimizers = ["adam", "sgd"]
    hparams = EasyDict(
        n_layers=np.random.choice(np.arange(3, 10)).item(),
        hidden_size=np.random.choice(np.arange(256, 513)).item(),
        dropout_p=np.random.uniform(0, 0.2),
        weight_decay=loguniform.rvs(1e-8, 1e-3).item(),
        lr=loguniform.rvs(1e-4, 1e-1).item(),
        # optimizer=optimizers[np.random.choice(len(optimizers))],
        optimizer="adam",
        batch_size=np.random.choice([64, 128, 256, 512]).item(),
        initialization=initializations[np.random.choice(len(initializations))],
    )
    return hparams


def init_model(model, hparams, verbose):
    if model == "comvex-linear":
        model = COMVEXLinear(
            in_features=[160, 2320, 2320, 170],
            **hparams,
        )
        model_info = summary(
            model,
            input_data=[[torch.rand(1, shape) for shape in [160, 2320, 2320, 170]]],
            verbose=verbose,
        )
    elif model == "comvex-conv":
        model = COMVEXConv(
            in_features=[10, 145, 145, 17],
            **hparams,
        )
        model_info = summary(
            model,
            input_data=[
                [
                    torch.rand(1, *shape)
                    for shape in [[1, 16, 10], [1, 16, 145], [1, 16, 145], [1, 10, 17]]
                ]
            ],
            verbose=verbose,
        )
    elif model in ["fc", "fc-stats"]:
        in_features = 4970 if model == "fc" else 56
        model = FullyConnected(
            in_features=in_features,
            **hparams,
        )
        model_info = summary(model, input_shape=(1, in_features), verbose=verbose)
    return model, model_info


def train(args):
    hparams = sample_hparams()
    if args.embedding_dim:
        hparams.embedding_dim = args.embedding_dim
    if args.dynamic_embedding_dim:
        hparams.dynamic_embedding_dim = args.dynamic_embedding_dim

    print(json.dumps(dict(hparams), indent=4))
    config = {**dict(hparams), **vars(args)}
    if args.wandb:
        wandb.init(
            project=args.project_name,
            entity="chrisliu298",
            config=config,
        )

    train_dataset, val_dataset, test_dataset = load_datasets(
        args.train_data_path, args.val_data_path, args.test_data_path, args.model
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
    # rand_idx = torch.randint(0, len(test_dataloader.dataset), size=(100,))
    # pred = torch.cat(
    #     trainer.predict(
    #         ckpt_path=model_checkpoint_callback.best_model_path,
    #         dataloaders=test_dataloader,
    #     )
    # )
    # print("True labels:")
    # y_test = test_dataset["test_acc"]
    # print(y_test[rand_idx].numpy())
    # print("Predictions:")
    # print(pred[rand_idx].numpy())
    if args.wandb:
        wandb.finish(quiet=True)

    # model.load_from_checkpoint(model_checkpoint_callback.best_model_path)
    # torch.save(model.state_dict(), f"{args.project_name}.pt")


def main():
    args = parse_args()
    args.num_workers = int(os.cpu_count() / 2)
    setup(args)
    train(args)


if __name__ == "__main__":
    main()
