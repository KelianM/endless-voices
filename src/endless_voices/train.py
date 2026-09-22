"""Fine-tune a causal language model using a small LoRA adapter."""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

from endless_voices.common import load_config, load_tokenizer
from endless_voices.data import ConversationCollator, tokenize_conversations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    training = TrainingArguments(
        **config["training"],
        report_to="none",
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
    )
    output = Path(training.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory {output} is not empty; choose a fresh directory.")
    set_seed(training.seed)
    source = config["model"]["name_or_path"]
    revision = config["model"].get("revision", "main")
    tokenizer = load_tokenizer(source, revision)
    dataset = tokenize_conversations(tokenizer=tokenizer, **config["data"])
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, torch_dtype=torch.float32
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, **config["lora"]))
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=ConversationCollator(tokenizer),
    )
    print(f"Training on {training.device} with {len(dataset)} conversations")
    trainer.train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved adapter and tokenizer to {output}")


if __name__ == "__main__":
    main()
