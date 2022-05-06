import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Model and training args
    parser.add_argument(
        "--model",
        type=str,
        default="comvex-conv",
        choices=["comvex-linear", "comvex-conv", "fc", "fc-stats"],
    )
    parser.add_argument("--embedding_dim", type=int)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=300)

    # Data args
    parser.add_argument("--train_data_path", type=str, default="train.pt")
    parser.add_argument("--val_data_path", type=str, default="val.pt")
    parser.add_argument("--test_data_path", type=str, default="test.pt")

    # Experiment args
    parser.add_argument("--project_name", type=str, default="comvex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()
