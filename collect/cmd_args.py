import argparse

parser = argparse.ArgumentParser()

# Experiment args
parser.add_argument("--project_name", type=str, default="cifar10")
parser.add_argument("--model_ckpt_path", type=str, default="models/")
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--verbose", type=int, default=0)

# Dataset args
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--hw", type=int, default=32)
parser.add_argument("--channels", type=int, default=3)

# Model and training args
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--model_name", type=str, default="cnn-16x32x64")
parser.add_argument("--output_size", type=int, default=10)


def parse_args():
    return parser.parse_args()
