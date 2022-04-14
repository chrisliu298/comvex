import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class Model(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc11 = nn.Linear(3 * 32 * 32 + 1, emb_dim)
        self.fc12 = nn.Linear(512 + 1, emb_dim)
        self.fc2 = nn.Linear(emb_dim * 2, 512)
        self.fc = nn.Linear(512, 1)

    def forward(self, w1, b1, w2, b2):
        out1 = torch.mean(self.fc11(self.make_input(w1, b1)), dim=0)
        out2 = torch.mean(self.fc12(self.make_input(w2, b2)), dim=0)
        print(out1.shape, out2.shape)
        concat = torch.cat([out1, out2])
        out = F.relu(self.fc2(concat))
        out = self.fc(out)
        return out

    def make_input(self, weight, bias):
        return torch.hstack([weight, bias.view(-1, 1)])


x = nn.Sequential(
    nn.Linear(3 * 32 * 32, 512),
    nn.Linear(512, 10),
)
inputs = list(x.state_dict().values())
model = Model(32)
summary(model, input_data=inputs)
