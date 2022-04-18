import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--project_name", type=str, default="cifar10")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--max_epochs", type=int, default=300)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--model_ckpt_path", type=str, default="models/")
parser.add_argument("--model_name", type=str, default="resnet-18")
parser.add_argument("--output_size", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb", action="store_true")

parser.add_argument("--hw", type=int, default=32)
parser.add_argument("--channels", type=int, default=3)


def parse_args():
    return parser.parse_args()
