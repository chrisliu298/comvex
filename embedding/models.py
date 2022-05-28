from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class COMVEXLinear(nn.Module):
    def __init__(
        self,
        dropout_p,
        hidden_size,
        n_layers,
        embedding_dim=64,
        in_features=[1792, 36928, 36928, 650],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features = copy(in_features)
        self.embedding_dim = embedding_dim
        # self.num_params = [448, 2320, 2320, 170]

        self._encoders = []
        self._layers = []
        for idx, in_feature in enumerate(self.in_features):
            encoder = nn.Linear(
                in_feature,
                self.embedding_dim,
            )
            self._encoders.append(encoder)
            self.add_module("linear_encoder{}".format(idx + 1), encoder)

        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(self.embedding_dim * len(self._encoders), hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            self._layers += [layer, nn.ReLU(), nn.Dropout(dropout_p)]

        self.features = nn.Sequential(*self._layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        encodings = []
        for x, encoder in zip(xs, self._encoders):
            encodings.append(encoder(x))
        x = torch.hstack(encodings)
        return encodings, self.sigmoid(self.fc(self.features(x))).squeeze()


class COMVEXConv(nn.Module):
    def __init__(
        self,
        dropout_p,
        hidden_size,
        n_layers,
        embedding_dim=64,
        in_features=[28, 577, 577, 65],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features = copy(in_features)
        self.embedding_dim = embedding_dim

        self._encoders = []
        self._layers = []
        for idx, in_feature in enumerate(self.in_features):
            encoder = nn.Conv2d(1, self.embedding_dim, (3, in_feature))
            self._encoders.append(encoder)
            self.add_module("conv_encoder{}".format(idx + 1), encoder)

        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(self.embedding_dim * len(self._encoders), hidden_size)
            else:
                layer = nn.Linear(hidden_size, hidden_size)
            self._layers += [layer, nn.ReLU(), nn.Dropout(dropout_p)]

        self.features = nn.Sequential(*self._layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        encodings = []
        for x, encoder in zip(xs, self._encoders):
            x = encoder(x).squeeze(3)
            encodings.append(F.max_pool1d(x, x.size(2)).squeeze(2))
        x = torch.hstack(encodings)
        return encodings, self.sigmoid(self.fc(self.features(x))).squeeze()


class FCNet_LinearL1(nn.Module):
    def __init__(
        self,
        dropout_p,
        hidden_size,
        n_layers,
        in_features,
        embedding_dim=64,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim * 4
        self._layers = []

        layer = nn.Linear(in_features, self.embedding_dim)
        self._layers += [layer]

        for i in range(n_layers):
            layer = (
                nn.Linear(self.embedding_dim, hidden_size)
                if i == 0
                else nn.Linear(hidden_size, hidden_size)
            )
            self._layers += [layer, nn.ReLU(), nn.Dropout(dropout_p)]

        self.features = nn.Sequential(*self._layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encodings = self.features[0](x)
        return encodings, self.sigmoid(self.fc(self.features[1:](encodings))).squeeze()
