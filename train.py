import argparse
import json
import os
from copy import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from torchmetrics.functional import r2_score


class Predictor(pl.LightningModule):
    def __init__(
        self,
        emb_dim,
        hidden_size,
        n_layers,
        dropout,
        lr,
        in_shapes=[448, 2320, 2320, 170],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_shapes = copy(in_shapes)
        self._layers = []
        self.fc11 = nn.Linear(self.in_shapes[0], emb_dim)
        self.fc12 = nn.Linear(self.in_shapes[1], emb_dim)
        self.fc13 = nn.Linear(self.in_shapes[2], emb_dim)
        self.fc14 = nn.Linear(self.in_shapes[3], emb_dim)

        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(emb_dim * 4, hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            self._layers.append(layer)
            self.add_module("fc{}".format(i + 1), layer)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, w1, w2, w3, w4):
        out1 = self.fc11(w1)
        out2 = self.fc12(w2)
        out3 = self.fc13(w3)
        out4 = self.fc14(w4)

        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        x = torch.hstack([out1, out2, out3, out4])
        # print(x.shape)

        batch_size = x.size(0)
        for layer in self._layers:
            x = self.dropout(F.relu(layer(x)))
        out = torch.sigmoid(self.fc(x))
        return out.view(batch_size)

    def evaluate(self, batch, stage=None):
        x1, x2, x3, x4, y = batch
        output = self(x1, x2, x3, x4)
        loss = F.mse_loss(output, y)
        r2 = r2_score(output, y).detach()
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_r2", r2, logger=True)
        return loss, r2

    def training_step(self, batch, batch_idx):
        loss, r2 = self.evaluate(batch, "train")
        return {"loss": loss, "train_r2": r2}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([i["loss"] for i in outputs]).mean()
        self.log("avg_train_loss", avg_loss, prog_bar=True, logger=True)
        avg_r2 = torch.stack([i["train_r2"] for i in outputs]).mean()
        self.log("avg_train_r2", avg_r2, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        loss, r2 = self.evaluate(batch, "val")
        return {"val_loss": loss, "val_r2": r2}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        self.log("avg_val_loss", avg_loss, prog_bar=True, logger=True)
        avg_r2 = torch.stack([i["val_r2"] for i in outputs]).mean()
        self.log("avg_val_r2", avg_r2, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, r2 = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_r2": r2}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        self.log("avg_test_loss", avg_loss, prog_bar=True, logger=True)
        avg_r2 = torch.stack([i["test_r2"] for i in outputs]).mean()
        self.log("avg_test_r2", avg_r2, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        x1, x2, x3, x4, _ = batch
        return self(x1, x2, x3, x4)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        return optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="ep-network")
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--hidden_size", type=int, default=512)
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--predictor_config", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.num_workers = int(os.cpu_count() / 2)
    setup(args)
    train(args)


def setup(args):
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, linewidth=60, sci_mode=False)


def train(args):
    hparams = EasyDict(
        n_layers=np.random.choice(np.arange(1, 7)).item(),
        hidden_size=np.random.choice(np.arange(128, 513)).item(),
        dropout=np.random.uniform(0, 0.5),
        lr=np.random.uniform(1e-3, 1e-1),
        batch_size=np.random.choice([32, 64, 128, 256]).item(),
        emb_dim=np.random.choice([16, 32, 64, 128]).item(),
    )
    if args.predictor_config != None:
        config = json.load(open(args.predictor_config))
        hparams = EasyDict(
            n_layers=config["n_layers"]["value"],
            hidden_size=config["hidden_size"]["value"],
            dropout=config["dropout"]["value"],
            lr=config["lr"]["value"],
            batch_size=config["batch_size"]["value"],
        )
    print(json.dumps(dict(hparams), indent=4))
    config = {**dict(hparams), **vars(args)}
    if args.wandb:
        wandb.init(
            project=args.project_name,
            entity="chrisliu298",
            config=config,
        )

    train = torch.load("train.pt")
    val = torch.load("val.pt")
    test = torch.load("test.pt")

    train_dataset = TensorDataset(
        train["w1"], train["w2"], train["w3"], train["w4"], train["test_acc"]
    )
    val_dataset = TensorDataset(
        val["w1"], val["w2"], val["w3"], val["w4"], val["test_acc"]
    )
    test_dataset = TensorDataset(
        test["w1"], test["w2"], test["w3"], test["w4"], test["test_acc"]
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = Predictor(
        emb_dim=hparams.emb_dim,
        hidden_size=hparams.hidden_size,
        n_layers=hparams.n_layers,
        dropout=hparams.dropout,
        lr=hparams.lr,
    )
    x1, x2, x3, x4, _ = next(iter(train_dataloader))
    info = summary(model, input_data=(x1, x2, x3, x4))
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
            project="gen-gap-prediction",
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

    model.load_from_checkpoint(model_checkpoint_callback.best_model_path)
    torch.save(model.state_dict(), f"{args.project_name}.pt")


if __name__ == "__main__":
    main()
