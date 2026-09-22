"""Chat with a base model, optionally loading a LoRA adapter."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, set_seed

from endless_voices.common import load_config, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.toml")
    parser.add_argument("--adapter", help="Local adapter directory or Hub adapter ID")
    parser.add_argument("--system", default="", help="Optional identity prompt")
    parser.add_argument("--system-file", type=Path, help="Read an identity prompt from a text file")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7, help="Zero for greedy decoding")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.system and args.system_file:
        parser.error("use either --system or --system-file")
    if args.temperature < 0 or not 0 < args.max_new_tokens < args.max_context_tokens:
        parser.error("require temperature >= 0 and 0 < max-new-tokens < max-context-tokens")
    config = load_config(args.config)
    source = config["model"]["name_or_path"]
    revision = config["model"].get("revision", "main")
    tokenizer = load_tokenizer(args.adapter or source, None if args.adapter else revision)
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, torch_dtype=torch.float32
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    model.to(device).eval()
    set_seed(args.seed)
    system = (
        args.system_file.read_text(encoding="utf-8").strip() if args.system_file else args.system
    )
    initial = [{"role": "system", "content": system}] if system else []
    messages = initial.copy()
    context_limit = min(
        args.max_context_tokens,
        getattr(model.config, "max_position_embeddings", args.max_context_tokens),
        tokenizer.model_max_length,
    )
    print(f"Ready on {device}. /reset clears history; /quit exits.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == "/quit":
            break
        if prompt == "/reset":
            messages = initial.copy()
            continue
        if not prompt:
            continue
        candidate = messages + [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            candidate, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        if inputs["input_ids"].shape[1] + args.max_new_tokens > context_limit:
            print("Context limit reached. Use /reset or shorten your message.")
            continue
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                **({"temperature": args.temperature} if args.temperature > 0 else {}),
                pad_token_id=tokenizer.pad_token_id,
            )
        reply = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        print(f"Voice: {reply}")
        messages = candidate + [{"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
