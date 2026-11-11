
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/kairo_train.jsonl"
OUTPUT_DIR = "output/kairo-lora"
MAX_LENGTH = 256


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    print("GPU:", torch.cuda.get_device_name(0))

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ValueError("Training dataset is empty.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    features = []

    for row in rows:
        parts = row["text"].split("Assistant:", 1)

        if len(parts) != 2:
            print("Skipping invalid example:", row["text"])
            continue

        question = parts[0].replace("User:", "", 1).strip()
        answer = parts[1].strip()

        if not question or not answer:
            continue

        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
        )

        full_ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )

        # Truncate long examples
        full_ids = full_ids[:MAX_LENGTH]
        prompt_length = min(len(prompt_ids), len(full_ids))

        # Calculate loss on the assistant answer only
        labels = list(full_ids)
        labels[:prompt_length] = [-100] * prompt_length

        if all(label == -100 for label in labels):
            continue

        features.append({
            "input_ids": list(full_ids),
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        })

    if not features:
        raise ValueError("No valid training examples found.")

    dataset = Dataset.from_list(features)
    print("Training examples:", len(dataset))

    # Pad input tokens and labels safely
    def collate_batch(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"] + [tokenizer.pad_token_id] * pad_len
            )
            attention_masks.append(
                item["attention_mask"] + [0] * pad_len
            )
            labels.append(
                item["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks, dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=1e-5,
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_batch,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("KAIRO LoRA adapter saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
