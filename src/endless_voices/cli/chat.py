"""Chat with a base model, optionally loading a LoRA adapter."""

import argparse
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="endless-voices chat", description=__doc__)
    parser.add_argument("--config", default="configs/train.toml")
    parser.add_argument("--adapter", help="Local adapter directory or Hub adapter ID")
    parser.add_argument("--system", default="", help="Optional identity prompt")
    parser.add_argument("--system-file", type=Path, help="Read an identity prompt from a text file")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7, help="Zero for greedy decoding")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.system and args.system_file:
        parser.error("use either --system or --system-file")
    if args.temperature < 0 or not 0 < args.max_new_tokens < args.max_context_tokens:
        parser.error("require temperature >= 0 and 0 < max-new-tokens < max-context-tokens")
    from endless_voices.chat import run_chat
    from endless_voices.common import load_config

    system = (
        args.system_file.read_text(encoding="utf-8").strip() if args.system_file else args.system
    )
    run_chat(
        load_config(args.config),
        adapter=args.adapter,
        system=system,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
