import math
import os
from copy import copy
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import accuracy


class BaseModel(pl.LightningModule):
    def __init__(self, save_path, seed, **kwargs):
        super().__init__()
        self.seed = seed
        self.save_path = save_path if "/" in save_path else save_path + "/"
        if not os.path.isdir(self.save_path):
            print("Creating new directory {}...".format(self.save_path))
            os.mkdir(self.save_path)

    def evaluate(self, batch, stage=None):
        x, y = batch
        output = self(x)
        pred = torch.argmax(output, dim=1)
        loss = F.cross_entropy(output, y)
        acc = accuracy(pred, y)
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_acc", acc, logger=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "train")
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, acc = self.evaluate(batch, "val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs) -> None:
        train_outputs, test_outputs = outputs
        avg_train_acc = torch.stack([i["val_acc"] for i in train_outputs]).mean()
        avg_train_loss = torch.stack([i["val_loss"] for i in train_outputs]).mean()
        avg_val_acc = torch.stack([i["val_acc"] for i in test_outputs]).mean()
        avg_val_loss = torch.stack([i["val_loss"] for i in test_outputs]).mean()

        self.log("avg_train_acc", avg_train_acc, logger=True)
        self.log("avg_train_loss", avg_train_loss, logger=True)
        self.log("avg_val_acc", avg_val_acc, logger=True)
        self.log("avg_val_loss", avg_val_loss, logger=True)

        prefix = datetime.now().strftime("%Y%m%d%H%M%S%f")
        checkpoint = {
            "seed": self.seed,
            "epoch": self.current_epoch,
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "train_loss": avg_train_loss.item(),
            "train_acc": avg_train_acc.item(),
            "test_loss": avg_val_loss.item(),
            "test_acc": avg_val_acc.item(),
        }
        torch.save(
            checkpoint,
            self.save_path + f"{prefix}_{self.current_epoch}.pt",
        )

    def test_step(self, batch, batch_idx, dataloader_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs) -> None:
        train_outputs, test_outputs = outputs
        avg_train_acc = torch.stack([i["test_acc"] for i in train_outputs]).mean()
        avg_train_loss = torch.stack([i["test_loss"] for i in train_outputs]).mean()
        avg_test_acc = torch.stack([i["test_acc"] for i in test_outputs]).mean()
        avg_test_loss = torch.stack([i["test_loss"] for i in test_outputs]).mean()

        self.log("final_avg_train_acc", avg_train_acc, logger=True)
        self.log("final_avg_test_acc", avg_test_acc, logger=True)
        self.log("final_avg_train_loss", avg_train_loss, logger=True)
        self.log("final_avg_test_loss", avg_test_loss, logger=True)

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        return optimizer

    def init_weights(self, weights, init_method):
        if init_method == "xavier":
            nn.init.xavier_uniform_(weights)
        elif init_method == "he":
            nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        elif init_method == "orthogonal":
            nn.init.orthogonal_(weights)
        elif init_method == "uniform":
            nn.init.uniform_(weights)
        elif init_method == "zeros":
            nn.init.zeros_(weights)
        else:
            raise ValueError(f"Invalid init method {init_method}")


class CNN(BaseModel):
    def __init__(
        self,
        n_units,
        optimizer,
        lr,
        weight_decay,
        dropout_p,
        initializer,
        activation,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._n_units = copy(n_units)
        self._layers = []
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p
        self.initializer = initializer
        self.activation = activation

        for i in range(1, len(n_units) - 1):
            layer = nn.Conv2d(n_units[i - 1], n_units[i], 3)
            self.init_weights(layer.weight.data, self.initializer)
            self.init_weights(layer.bias.data, "zeros")
            activation_layer = nn.ReLU() if self.activation == "relu" else nn.Tanh()
            self._layers += [layer, activation_layer, nn.Dropout(self.dropout_p)]

        self._layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
        self.features = nn.Sequential(*self._layers)
        self.fc = nn.Linear(n_units[-2], n_units[-1])
        self.init_weights(layer.weight.data, self.initializer)
        self.init_weights(layer.bias.data, "zeros")

    def forward(self, x):
        return self.fc(self.features(x))
