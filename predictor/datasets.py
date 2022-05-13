import torch
from torch.utils.data import TensorDataset


def load_datasets(train_data_path, val_data_path, test_data_path, model, hidden_sizes):
    train = torch.load(train_data_path)
    val = torch.load(val_data_path)
    test = torch.load(test_data_path)
    train_size = len(train["test_acc"])
    val_size = len(val["test_acc"])
    test_size = len(test["test_acc"])
    print(train_size, val_size, test_size)

    if model == "comvex-linear":
        train_dataset = TensorDataset(
            *[train[f"w{i + 1}"] for i in range(4)], train["test_acc"]
        )
        val_dataset = TensorDataset(
            *[val[f"w{i + 1}"] for i in range(4)], val["test_acc"]
        )
        test_dataset = TensorDataset(
            *[test[f"w{i + 1}"] for i in range(4)], test["test_acc"]
        )

    elif model == "comvex-conv":
        train_dataset = TensorDataset(
            *[
                train[f"w{i + 1}"].view(train_size, 1, h, -1)
                for i, h in zip(range(4), hidden_sizes)
            ],
            train["test_acc"],
        )
        val_dataset = TensorDataset(
            *[
                val[f"w{i + 1}"].view(val_size, 1, h, -1)
                for i, h in zip(range(4), hidden_sizes)
            ],
            val["test_acc"],
        )
        test_dataset = TensorDataset(
            *[
                test[f"w{i + 1}"].view(test_size, 1, h, -1)
                for i, h in zip(range(4), hidden_sizes)
            ],
            test["test_acc"],
        )
    elif model == "fc":
        train_dataset = TensorDataset(
            torch.cat([train[f"w{i + 1}"] for i in range(4)], dim=1),
            train["test_acc"],
        )
        val_dataset = TensorDataset(
            torch.cat([val[f"w{i + 1}"] for i in range(4)], dim=1),
            val["test_acc"],
        )
        test_dataset = TensorDataset(
            torch.cat([test[f"w{i + 1}"] for i in range(4)], dim=1),
            test["test_acc"],
        )
    elif model == "fc-stats":
        train_dataset = TensorDataset(train["stats"], train["test_acc"])
        val_dataset = TensorDataset(val["stats"], val["test_acc"])
        test_dataset = TensorDataset(test["stats"], test["test_acc"])
    else:
        raise ValueError(f"Model {model} does not exist.")

    return (train_dataset, val_dataset, test_dataset)
