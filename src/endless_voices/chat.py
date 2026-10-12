"""Chat with a base model, optionally loading a LoRA adapter."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, set_seed

from endless_voices.common import load_tokenizer


def run_chat(
    config: dict,
    *,
    adapter: str | None = None,
    system: str = "",
    device: str = "auto",
    max_new_tokens: int = 128,
    max_context_tokens: int = 2048,
    temperature: float = 0.7,
    seed: int = 42,
) -> None:
    """Run an interactive chat with a base model and optional LoRA adapter."""
    source = config["model"]["name_or_path"]
    revision = config["model"].get("revision", "main")
    tokenizer = load_tokenizer(adapter or source, None if adapter else revision)
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, torch_dtype=torch.float32
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    model.to(device).eval()
    set_seed(seed)
    initial = [{"role": "system", "content": system}] if system else []
    messages = initial.copy()
    context_limit = min(
        max_context_tokens,
        getattr(model.config, "max_position_embeddings", max_context_tokens),
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
        if inputs["input_ids"].shape[1] + max_new_tokens > context_limit:
            print("Context limit reached. Use /reset or shorten your message.")
            continue
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature} if temperature > 0 else {}),
                pad_token_id=tokenizer.pad_token_id,
            )
        reply = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        print(f"Voice: {reply}")
        messages = candidate + [{"role": "assistant", "content": reply}]
