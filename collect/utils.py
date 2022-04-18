import torch


def get_weights(parameters):
    return torch.cat([param.flatten() for param in parameters]).to("cuda")


def get_stats(parameters):
    qs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to("cuda")
    stats = []
    for param in parameters:
        param = param.float()
        stats.append(
            torch.cat(
                [torch.mean(param).unsqueeze(0), torch.var(param).unsqueeze(0)]
            ).to("cuda")
        )
        stats.append(torch.quantile(param.flatten(), q=qs).to("cuda"))
    return torch.cat(stats).to("cuda")
