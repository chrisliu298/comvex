import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class Model(nn.Module):
    def __init__(self, emb_dim, hidden_size, in_shapes=[12560, 272, 272, 170]):
        super().__init__()
        self.in_shapes = in_shapes
        self.fc11 = nn.Linear(in_shapes[0], emb_dim)
        self.fc12 = nn.Linear(in_shapes[1], emb_dim)
        self.fc13 = nn.Linear(in_shapes[2], emb_dim)
        self.fc14 = nn.Linear(in_shapes[3], emb_dim)
        self.fc2 = nn.Linear(emb_dim * 4, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, w1, w2, w3, w4):
        # w1, b1, w2, b2, w3, b3, w4, b4 = wb
        out1 = self.fc11(w1)
        out2 = self.fc12(w2)
        out3 = self.fc13(w3)
        out4 = self.fc14(w4)

        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        concat = torch.hstack([out1, out2, out3, out4])
        # print(concat.shape)
        out = F.relu(self.fc2(concat))
        out = torch.sigmoid(self.fc(out))
        return out


x = nn.Sequential(
    nn.Linear(3 * 32 * 32, 512),
    nn.Linear(512, 10),
)
inputs = list(x.state_dict().values())
model = Model(32)
summary(model, input_data=inputs)
