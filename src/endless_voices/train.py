"""Fine-tune a causal language model using a small LoRA adapter."""

import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

from endless_voices.artifacts import file_hash
from endless_voices.common import load_tokenizer
from endless_voices.context import TokenizerCounter
from endless_voices.context_generation import open_database
from endless_voices.data import ConversationCollator, chat_ids, tokenize_conversations
from endless_voices.dataset.storage import SceneDataset


def run_training(config: dict) -> None:
    """Train and save a LoRA adapter using the supplied configuration."""
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
    data = config["data"]
    prepared = SceneDataset.load(data["path"], data.get("split", "train"))
    context = config["context"]
    counter = TokenizerCounter(tokenizer)
    prepared_contexts = []
    session = None
    with open_database(context) as database:
        for index, record in enumerate(prepared):
            session = prepared.retrieval_session(index, database)
            target = record["messages"][-1]
            bare = session.initial_messages()
            target_tokens = len(chat_ids([*bare, target], tokenizer)) - counter.messages(bare)
            budget = min(context.get("max_input_tokens", 8192), data["max_length"] - target_tokens)
            messages = session.static_messages(counter, budget, context.get("depth", 4))
            prepared_contexts.append({
                "sample_id": record["metadata"]["id"], "messages": [*messages, target],
                "retrieval_trace": session.trace,
            })
    del database, session
    dataset = tokenize_conversations(
        [row["messages"] for row in prepared_contexts], tokenizer, data["max_length"],
        data.get("loss", "continuation"))
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
    (output / "training_contexts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared_contexts))
    run_config = {**config, "context_index_sha256": file_hash(context["index"])}
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(f"Saved adapter and tokenizer to {output}")
