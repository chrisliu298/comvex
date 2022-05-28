import torch
from torch.utils.data import TensorDataset


def load_dataset(dataset_path, model, hidden_sizes):
    raw_dataset = torch.load(dataset_path)
    dataset_size = len(raw_dataset["test_acc"])
    print(f"{dataset_path} size: {dataset_size}")
    if model == "comvex-linear":
        dataset = TensorDataset(
            *[raw_dataset[f"w{i + 1}"] for i in range(4)], raw_dataset["test_acc"]
        )
    elif model == "comvex-conv":
        dataset = TensorDataset(
            *[
                raw_dataset[f"w{i + 1}"].view(dataset_size, 1, h, -1)
                for i, h in zip(range(4), hidden_sizes)
            ],
            raw_dataset["test_acc"],
        )
    elif model == "fc" or model == "fc-linear":
        dataset = TensorDataset(
            torch.cat([raw_dataset[f"w{i + 1}"] for i in range(4)], dim=1),
            raw_dataset["test_acc"],
        )
    else:
        raise ValueError(f"Model {model} does not exist.")
    del raw_dataset
    return dataset


# def load_datasets(
#     train_dataset_path, val_dataset_path, test_dataset_path, model, hidden_sizes
# ):
#     train = torch.load(train_dataset_path)
#     val = torch.load(val_dataset_path)
#     test = torch.load(test_dataset_path)
#     train_size = len(train["test_acc"])
#     val_size = len(val["test_acc"])
#     test_size = len(test["test_acc"])
#     print(train_size, val_size, test_size)

#     if model == "comvex-linear":
#         train_dataset = TensorDataset(
#             *[train[f"w{i + 1}"] for i in range(4)], train["test_acc"]
#         )
#         val_dataset = TensorDataset(
#             *[val[f"w{i + 1}"] for i in range(4)], val["test_acc"]
#         )
#         test_dataset = TensorDataset(
#             *[test[f"w{i + 1}"] for i in range(4)], test["test_acc"]
#         )

#     elif model == "comvex-conv":
#         train_dataset = TensorDataset(
#             *[
#                 train[f"w{i + 1}"].view(train_size, 1, h, -1)
#                 for i, h in zip(range(4), hidden_sizes)
#             ],
#             train["test_acc"],
#         )
#         val_dataset = TensorDataset(
#             *[
#                 val[f"w{i + 1}"].view(val_size, 1, h, -1)
#                 for i, h in zip(range(4), hidden_sizes)
#             ],
#             val["test_acc"],
#         )
#         test_dataset = TensorDataset(
#             *[
#                 test[f"w{i + 1}"].view(test_size, 1, h, -1)
#                 for i, h in zip(range(4), hidden_sizes)
#             ],
#             test["test_acc"],
#         )
#     elif model == "fc":
#         train_dataset = TensorDataset(
#             torch.cat([train[f"w{i + 1}"] for i in range(4)], dim=1),
#             train["test_acc"],
#         )
#         val_dataset = TensorDataset(
#             torch.cat([val[f"w{i + 1}"] for i in range(4)], dim=1),
#             val["test_acc"],
#         )
#         test_dataset = TensorDataset(
#             torch.cat([test[f"w{i + 1}"] for i in range(4)], dim=1),
#             test["test_acc"],
#         )
#     elif model == "fc-stats":
#         train_dataset = TensorDataset(train["stats"], train["test_acc"])
#         val_dataset = TensorDataset(val["stats"], val["test_acc"])
#         test_dataset = TensorDataset(test["stats"], test["test_acc"])
#     else:
#         raise ValueError(f"Model {model} does not exist.")
#     del train
#     del val
#     del test
#     return (train_dataset, val_dataset, test_dataset)
