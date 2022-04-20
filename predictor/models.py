from copy import copy
from math import ceil

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import r2_score


class Predictor(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, r2 = self.evaluate(batch, "train")
        return {"loss": loss, "train_r2": r2}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([i["loss"] for i in outputs]).mean()
        avg_r2 = torch.stack([i["train_r2"] for i in outputs]).mean()
        self.log("avg_train_loss", avg_loss, prog_bar=True, logger=True)
        self.log("avg_train_r2", avg_r2, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        loss, r2 = self.evaluate(batch, "val")
        return {"val_loss": loss, "val_r2": r2}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        avg_r2 = torch.stack([i["val_r2"] for i in outputs]).mean()
        self.log("avg_val_loss", avg_loss, prog_bar=True, logger=True)
        self.log("avg_val_r2", avg_r2, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, r2 = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_r2": r2}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        avg_r2 = torch.stack([i["test_r2"] for i in outputs]).mean()
        self.log("avg_test_loss", avg_loss, prog_bar=True, logger=True)
        self.log("avg_test_r2", avg_r2, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        return optimizer


class EPNetwork(Predictor):
    def __init__(
        self,
        hidden_size,
        n_layers,
        dropout,
        emb_dim,
        in_features=[448, 2320, 2320, 170],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.in_features = copy(in_features)
        self._layers = []
        # self.dynamic_emb_dim = [ceil(i / 10) for i in self.in_features]
        # self.fc11 = nn.Linear(self.in_features[0], self.dynamic_emb_dim[0])
        # self.fc12 = nn.Linear(self.in_features[1], self.dynamic_emb_dim[1])
        # self.fc13 = nn.Linear(self.in_features[2], self.dynamic_emb_dim[2])
        # self.fc14 = nn.Linear(self.in_features[3], self.dynamic_emb_dim[3])
        self.fc11 = nn.Linear(self.in_features[0], emb_dim)
        self.fc12 = nn.Linear(self.in_features[1], emb_dim)
        self.fc13 = nn.Linear(self.in_features[2], emb_dim)
        self.fc14 = nn.Linear(self.in_features[3], emb_dim)

        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(emb_dim, hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            self._layers.append(layer)
            self.add_module("fc{}".format(i + 2), layer)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x1, x2, x3, x4):
        batch_size = x1.size(0)
        x1 = self.fc11(x1)
        x2 = self.fc12(x2)
        x3 = self.fc13(x3)
        x4 = self.fc14(x4)

        x = torch.hstack([x1, x2, x3, x4]).view(batch_size, 4, -1)
        x = torch.mean(x, dim=1)

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

    def predict_step(self, batch, batch_idx):
        x1, x2, x3, x4, _ = batch
        return self(x1, x2, x3, x4)


class FCNetwork(Predictor):
    def __init__(
        self,
        hidden_size,
        n_layers,
        dropout,
        in_features,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._layers = []

        for i in range(n_layers + 1):
            if i == 0:
                layer = nn.Linear(in_features, hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            self._layers.append(layer)
            self.add_module("fc{}".format(i + 1), layer)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        for layer in self._layers:
            x = self.dropout(F.relu(layer(x)))
        out = torch.sigmoid(self.fc(x))
        return out.view(batch_size)

    def evaluate(self, batch, stage=None):
        x, y = batch
        output = self(x)
        loss = F.mse_loss(output, y)
        r2 = r2_score(output, y).detach()
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_r2", r2, logger=True)
        return loss, r2

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)
