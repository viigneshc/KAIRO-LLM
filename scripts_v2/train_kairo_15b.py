
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Model and dataset settings
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_PATH = Path("data_v2/kairo_bilingual_v2.jsonl")
OUTPUT_DIR = Path("output_v2/kairo-1.5b-tanglish-v2")

MAX_LENGTH = 512


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run scripts_v2/improve_dataset.py first."
        )

    with DATA_PATH.open("r", encoding="utf-8-sig") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    valid_rows = []

    for index, row in enumerate(rows, start=1):
        messages = row.get("messages", [])

        if len(messages) != 3:
            print(f"Skipping invalid record #{index}")
            continue

        if not all(
            isinstance(message.get("content"), str)
            and message["content"].strip()
            for message in messages
        ):
            print(f"Skipping empty record #{index}")
            continue

        valid_rows.append(row)

    if not valid_rows:
        raise ValueError("No valid training examples found.")

    print("Dataset records:", len(rows))
    print("Valid records:", len(valid_rows))

    return valid_rows


def prepare_features(rows, tokenizer):
    features = []

    for row in rows:
        messages = row["messages"]

        # Prompt contains system instructions and user question.
        prompt_ids = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=True,
            add_generation_prompt=True,
        )

        # Full conversation contains the expected assistant answer.
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Skip examples that exceed the context limit.
        # Avoid truncating away the assistant answer.
        if len(full_ids) > MAX_LENGTH:
            print(
                "Skipping long example:",
                messages[1]["content"][:60],
            )
            continue

        prompt_length = len(prompt_ids)

        if prompt_length >= len(full_ids):
            print(
                "Skipping example without answer tokens:",
                messages[1]["content"][:60],
            )
            continue

        labels = list(full_ids)

        # Train on the assistant answer, not the prompt.
        labels[:prompt_length] = [-100] * prompt_length

        features.append(
            {
                "input_ids": list(full_ids),
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )

    if not features:
        raise ValueError(
            "No usable examples after tokenization. "
            "Check the dataset and MAX_LENGTH."
        )

    print("Prepared training examples:", len(features))

    return Dataset.from_list(features)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch installation."
        )

    print("=" * 55)
    print("KAIRO - Bilingual Tanglish QLoRA Training")
    print("=" * 55)

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2,
        ),
        "GB",
    )

    rows = load_dataset()

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("\nLoading base model in 4-bit mode...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    base_model.config.use_cache = False

    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print("\nPreparing training data...")
    train_dataset = prepare_features(rows, tokenizer)

    def collate_batch(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"]
                + [tokenizer.pad_token_id] * pad_len
            )

            attention_masks.append(
                item["attention_mask"] + [0] * pad_len
            )

            labels.append(
                item["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(
                labels, dtype=torch.long
            ),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        learning_rate=5e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,

        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        optim="paged_adamw_8bit",

        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,

        report_to="none",
        dataloader_pin_memory=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_batch,
    )

    print("\nStarting training...")
    print("Base model:", MODEL_NAME)
    print("Dataset:", DATA_PATH)
    print("Output:", OUTPUT_DIR)
    print("Epochs:", training_args.num_train_epochs)

    trainer.train()

    print("\nSaving adapter and tokenizer...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\nTraining completed.")
    print("New adapter saved at:", OUTPUT_DIR)
    print("Original adapter remains at: output/kairo-lora")
    print("Previous 1.5B adapter remains at: output_v2/kairo-1.5b-qlora")


if __name__ == "__main__":
    main()
