
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = ROOT / "data_v8" / "kairo_bilingual_v8.jsonl"
OUTPUT_PATH = ROOT / "output_v9" / "kairo-1.5b-tanglish-v9"

MAX_LENGTH = 512
EPOCHS = 3
LEARNING_RATE = 2e-5

SYSTEM_PROMPT = (
    "You are KAIRO, a helpful and accurate AI assistant. "
    "Answer English questions in clear English. "
    "For Tanglish requests, use natural Tamil written only in English letters. "
    "Never use Tamil script in Tanglish answers. "
    "Follow the requested language, length, and format. "
    "Do not add code unless requested. "
    "Answer directly and avoid repetition. Never invent facts."
)


# ---------------- DATA LOADING ----------------

def load_records():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    records = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            messages = record.get("messages")

            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Line {line_number}: missing messages list."
                )

            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"Line {line_number}: final message must be assistant."
                )

            if any(
                message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(
                    f"Line {line_number}: invalid role or content."
                )

            records.append(record)

    if not records:
        raise ValueError("Dataset is empty.")

    return records


# ---------------- TRAINING ----------------

def main():
    # Never overwrite an existing V9 run.
    if OUTPUT_PATH.exists() and any(OUTPUT_PATH.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {OUTPUT_PATH}\n"
            "Existing files will not be overwritten."
        )

    records = load_records()
    print("Dataset records:", len(records))

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    def tokenize_record(record):
        messages = [dict(message) for message in record["messages"]]

        # Use one consistent system prompt.
        if messages[0]["role"] == "system":
            messages[0]["content"] = SYSTEM_PROMPT
        else:
            messages.insert(
                0,
                {"role": "system", "content": SYSTEM_PROMPT},
            )

        # Tokenize the entire conversation in one pass.
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        labels = [-100] * len(input_ids)

        # Find assistant-message spans using rendered conversation prefixes.
        # Verify each prefix matches the full sequence before masking.
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue

            prefix_before = tokenizer.apply_chat_template(
                messages[:index],
                tokenize=True,
                add_generation_prompt=False,
            )

            prefix_through = tokenizer.apply_chat_template(
                messages[: index + 1],
                tokenize=True,
                add_generation_prompt=False,
            )

            start = len(prefix_before)
            end = len(prefix_through)

            if input_ids[:start] != prefix_before:
                raise ValueError(
                    "Chat-template prefix mismatch before assistant message. "
                    "Training stopped to prevent incorrect masking."
                )

            if input_ids[:end] != prefix_through:
                raise ValueError(
                    "Chat-template prefix mismatch inside assistant message. "
                    "Training stopped to prevent incorrect masking."
                )

            # Supervise assistant message tokens only.
            labels[start:end] = input_ids[start:end]

        # Truncate the sequence and its labels together.
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

        assistant_token_count = sum(
            token != -100 for token in labels
        )

        if assistant_token_count == 0:
            raise ValueError(
                "No assistant tokens remain after truncation. "
                "Check the dataset or increase MAX_LENGTH."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    print("Tokenizing dataset and validating assistant masks...")

    dataset = Dataset.from_list(records)

    tokenized_dataset = dataset.map(
        tokenize_record,
        remove_columns=dataset.column_names,
        desc="Tokenizing and masking",
    )

    # Validate every record before loading the GPU model.
    for index in range(len(tokenized_dataset)):
        example = tokenized_dataset[index]

        if len(example["input_ids"]) != len(example["labels"]):
            raise ValueError(
                f"Input/label length mismatch at record {index}."
            )

        if not any(label != -100 for label in example["labels"]):
            raise ValueError(
                f"No supervised assistant tokens at record {index}."
            )

    sample = tokenized_dataset[0]
    print("Mask validation passed.")
    print("First sample tokens:", len(sample["input_ids"]))
    print(
        "First sample assistant tokens:",
        sum(label != -100 for label in sample["labels"]),
    )

    # Ensure the output parent exists only after data validation.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading 4-bit base model...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_PATH / "checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    print("\nKAIRO V9 training starting...")
    trainer.train()

    model.save_pretrained(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    print("\nKAIRO V9 training completed.")
    print("Adapter saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
