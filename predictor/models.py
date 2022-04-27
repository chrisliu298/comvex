from copy import copy
from math import ceil

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import r2_score


class Predictor(pl.LightningModule):
    def __init__(self, **kwargs):
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
            nn.init.xavier_normal_(weights)
        elif init_method == "he":
            nn.init.kaiming_normal_(weights)
        elif init_method == "orthogonal":
            nn.init.orthogonal_(weights)
        elif init_method == "normal":
            nn.init.normal_(weights)
        elif init_method == "zeros":
            nn.init.zeros_(weights)
        else:
            raise ValueError(f"Invalid init method {init_method}")


class COMVEXLinear(Predictor):
    def __init__(
        self,
        optimizer,
        lr,
        weight_decay,
        dropout_p,
        initializer,
        hidden_size,
        n_layers,
        embedding_dim=64,
        dynamic_embedding_dim=False,
        in_features=[1792, 36928, 36928, 650],
        target_dim=[448, 2320, 2320, 170],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.in_features = copy(in_features)
        self.embedding_dim = embedding_dim
        self.dynamic_embedding_dim = dynamic_embedding_dim
        # self.num_params = [448, 2320, 2320, 170]
        self.target_dim = target_dim

        self._encoders = []
        self._layers = []
        for idx, in_feature in enumerate(self.in_features):
            self.embedding_dim = (
                self.target_dim[idx]
                if self.dynamic_embedding_dim
                else self.embedding_dim
            )
            net = nn.Linear(
                in_feature,
                self.embedding_dim,
            )
            self.init_weights(net.weight.data, self.initializer)
            self.init_weights(net.bias.data, "zeros")
            self._encoders.append(net)
            self.add_module("encoder{}".format(idx + 1), net)

        for i in range(n_layers):
            if i == 0:
                in_features = (
                    sum(self.target_dim)
                    if self.dynamic_embedding_dim
                    else self.embedding_dim * len(self._encoders)
                )

                layer = nn.Linear(in_features, hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            self.init_weights(layer.weight.data, self.initializer)
            self.init_weights(layer.bias.data, "zeros")
            self._layers.append(layer)
            self.add_module("fc{}".format(i + 2), layer)

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, 1)
        self.init_weights(self.fc.weight.data, self.initializer)
        self.init_weights(self.fc.bias.data, "zeros")

    def forward(self, xs):
        encodings = []
        for x, net in zip(xs, self._encoders):
            encodings.append(net(x))

        x = torch.hstack(encodings)

        for layer in self._layers:
            x = self.dropout(F.relu(layer(x)))
        out = torch.sigmoid(self.fc(x))
        return out.squeeze()

    def evaluate(self, batch, stage=None):
        x1, x2, x3, x4, y = batch
        output = self([x1, x2, x3, x4])
        loss = F.mse_loss(output, y)
        r2 = r2_score(output, y).detach()
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_r2", r2, logger=True)
        return loss, r2

    def predict_step(self, batch, batch_idx):
        x1, x2, x3, x4, _ = batch
        return self([x1, x2, x3, x4])


class COMVEXConv(Predictor):
    def __init__(
        self,
        optimizer,
        lr,
        weight_decay,
        dropout_p,
        initializer,
        hidden_size,
        n_layers,
        embedding_dim=64,
        dynamic_embedding_dim=False,
        in_features=[28, 577, 577, 65],
        target_dim=[448, 2320, 2320, 170],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.in_features = copy(in_features)
        self.embedding_dim = embedding_dim
        self.dynamic_embedding_dim = dynamic_embedding_dim
        self.target_dim = target_dim

        self._encoders = []
        self._layers = []
        for idx, in_feature in enumerate(self.in_features):
            # if self.dynamic_embedding_dim:
            #     net = nn.Conv2d(1, embedding_dim, (3, in_feature))
            # else:
            self.embedding_dim = (
                self.target_dim[idx]
                if self.dynamic_embedding_dim
                else self.embedding_dim
            )
            net = nn.Conv2d(1, self.embedding_dim, (3, in_feature))
            self.init_weights(net.weight.data, self.initializer)
            self.init_weights(net.bias.data, "zeros")
            self._encoders.append(net)
            self.add_module("conv_encoder{}".format(idx + 1), net)

        for i in range(n_layers):
            if i == 0:
                in_features = (
                    sum(self.target_dim)
                    if self.dynamic_embedding_dim
                    else self.embedding_dim * len(self._encoders)
                )
                layer = nn.Linear(in_features, hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            self.init_weights(layer.weight.data, self.initializer)
            self.init_weights(layer.bias.data, "zeros")
            self._layers.append(layer)
            self.add_module("fc{}".format(i + 2), layer)

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, 1)
        self.init_weights(self.fc.weight.data, self.initializer)
        self.init_weights(self.fc.bias.data, "zeros")

    def forward(self, xs):
        encodings = []
        for x, net in zip(xs, self._encoders):
            x = net(x).squeeze(3)
            encodings.append(F.max_pool1d(x, x.size(2)).squeeze(2))

        x = torch.hstack(encodings)

        for layer in self._layers:
            x = self.dropout(F.relu(layer(x)))
        out = torch.sigmoid(self.fc(x))
        return out.squeeze()

    def evaluate(self, batch, stage=None):
        x1, x2, x3, x4, y = batch
        output = self([x1, x2, x3, x4])
        loss = F.mse_loss(output, y)
        r2 = r2_score(output, y).detach()
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_r2", r2, logger=True)
        return loss, r2

    def predict_step(self, batch, batch_idx):
        x1, x2, x3, x4, _ = batch
        return self([x1, x2, x3, x4])


class FCNet(Predictor):
    def __init__(
        self,
        optimizer,
        lr,
        weight_decay,
        dropout_p,
        initializer,
        hidden_size,
        n_layers,
        embedding_dim,
        in_features,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.initializer = initializer
        self.weight_decay = weight_decay
        self._layers = []

        self.encoder = nn.Linear(in_features, embedding_dim)
        self.init_weights(self.encoder.weight.data, self.initializer)
        self.init_weights(self.encoder.bias.data, "zeros")
        # self._layers.append(layer)
        # self.add_module("fc1", layer)

        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(embedding_dim, hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            self.init_weights(layer.weight.data, self.initializer)
            self.init_weights(layer.bias.data, "zeros")
            self._layers.append(layer)
            self.add_module("fc{}".format(i + 2), layer)

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, 1)
        self.init_weights(self.fc.weight.data, self.initializer)
        self.init_weights(self.fc.bias.data, "zeros")

    def forward(self, x):
        x = self.encoder(x)
        for layer in self._layers:
            x = self.dropout(F.relu(layer(x)))
        out = torch.sigmoid(self.fc(x))
        return out.squeeze()

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
