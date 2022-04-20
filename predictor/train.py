import argparse
import json
import os

import numpy as np
import torch
import wandb
from easydict import EasyDict
from models import EPNetwork, FCNetwork
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="ep-network")
    parser.add_argument(
        "--model_arch", type=str, default="ep", choices=["ep", "fc-w", "fc-s"]
    )
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=1000)
    # parser.add_argument("--predictor_config", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def setup(args):
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, linewidth=60, sci_mode=False)


def train(args):
    hparams = EasyDict(
        n_layers=np.random.choice(np.arange(1, 5)).item()
        if args.n_layers == None
        else args.n_layers,
        hidden_size=np.random.choice(np.arange(128, 513)).item()
        if args.hidden_size == None
        else args.hidden_size,
        dropout=np.random.uniform(0, 0.5) if args.dropout == None else args.dropout,
        lr=np.random.uniform(1e-4, 1e-2) if args.lr == None else args.lr,
        batch_size=np.random.choice([32, 64, 128, 256]).item()
        if args.batch_size == None
        else args.batch_size,
    )
    if args.model_arch == "ep":
        hparams.emb_dim = (
            np.random.choice(np.arange(32, 129)).item()
            if args.emb_dim == None
            else args.emb_dim
        )
    # if args.predictor_config != None:
    #     config = json.load(open(args.predictor_config))
    #     hparams = EasyDict(
    #         n_layers=config["n_layers"]["value"],
    #         hidden_size=config["hidden_size"]["value"],
    #         dropout=config["dropout"]["value"],
    #         lr=config["lr"]["value"],
    #         batch_size=config["batch_size"]["value"],
    #     )
    print(json.dumps(dict(hparams), indent=4))
    config = {**dict(hparams), **vars(args)}
    print(config)
    if args.wandb:
        wandb.init(
            project=args.project_name,
            entity="chrisliu298",
            config=config,
        )

    train = torch.load("train.pt")
    val = torch.load("val.pt")
    test = torch.load("test.pt")

    if args.model_arch == "ep":
        train_dataset = TensorDataset(
            train["w1"], train["w2"], train["w3"], train["w4"], train["test_acc"]
        )
        val_dataset = TensorDataset(
            val["w1"], val["w2"], val["w3"], val["w4"], val["test_acc"]
        )
        test_dataset = TensorDataset(
            test["w1"], test["w2"], test["w3"], test["w4"], test["test_acc"]
        )
    elif args.model_arch == "fc-w":
        train_dataset = TensorDataset(
            torch.cat([train["w1"], train["w2"], train["w3"], train["w4"]], dim=1),
            train["test_acc"],
        )
        val_dataset = TensorDataset(
            torch.cat([val["w1"], val["w2"], val["w3"], val["w4"]], dim=1),
            val["test_acc"],
        )
        test_dataset = TensorDataset(
            torch.cat([test["w1"], test["w2"], test["w3"], test["w4"]], dim=1),
            test["test_acc"],
        )
        in_features = train_dataset[0][0].size(0)
    elif args.model_arch == "fc-s":
        train_dataset = TensorDataset(train["stats"], train["test_acc"])
        val_dataset = TensorDataset(val["stats"], val["test_acc"])
        test_dataset = TensorDataset(test["stats"], test["test_acc"])
        in_features = train_dataset[0][0].size(0)

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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.model_arch == "ep":
        model = EPNetwork(
            hidden_size=hparams.hidden_size,
            n_layers=hparams.n_layers,
            dropout=hparams.dropout,
            lr=hparams.lr,
            emb_dim=hparams.emb_dim,
        )
        x1, x2, x3, x4, _ = next(iter(train_dataloader))
        info = summary(model, input_data=(x1, x2, x3, x4))
    elif args.model_arch in ["fc-w", "fc-s"]:
        model = FCNetwork(
            hidden_size=hparams.hidden_size,
            n_layers=hparams.n_layers,
            dropout=hparams.dropout,
            in_features=in_features,
            lr=hparams.lr,
        )
        x, _ = next(iter(train_dataloader))
        info = summary(model, input_data=x)

    if args.wandb:
        wandb.log({"total_params": info.total_params})

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
    )
    trainer.test(
        ckpt_path=model_checkpoint_callback.best_model_path,
        dataloaders=test_dataloader,
    )
    rand_idx = torch.randint(0, len(test_dataloader.dataset), size=(100,))
    pred = torch.cat(
        trainer.predict(
            ckpt_path=model_checkpoint_callback.best_model_path,
            dataloaders=test_dataloader,
        )
    )
    print("True labels:")
    y_test = test["test_acc"]
    print(y_test[rand_idx].numpy())
    print("Predictions:")
    print(pred[rand_idx].numpy())
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
