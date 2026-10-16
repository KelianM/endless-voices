"""Fine-tune a causal language model using a small LoRA adapter."""

import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="endless-voices train", description=__doc__)
    parser.add_argument("--config", default="configs/train.toml")
    args = parser.parse_args(argv)
    from endless_voices.common import load_config
    from endless_voices.train import run_training

    run_training(load_config(args.config))
