
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
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = Path("data_v4/kairo_bilingual_v4.jsonl")
OUTPUT_DIR = Path("output_v4/kairo-1.5b-tanglish-v4")

MAX_LENGTH = 512


def load_dataset_records():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    records = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            messages = record.get("messages")

            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(
                    f"Invalid conversation at line {line_number}"
                )

            if (
                messages[0].get("role") != "system"
                or messages[-2].get("role") != "user"
                or messages[-1].get("role") != "assistant"
            ):
                raise ValueError(
                    f"Expected system/user/assistant messages "
                    f"at line {line_number}"
                )

            if not messages[-2].get("content") or not messages[-1].get("content"):
                raise ValueError(
                    f"Missing question or answer at line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError("Dataset is empty.")

    print("Dataset records loaded:", len(records))
    return records


def prepare_examples(records, tokenizer):
    prepared = []
    skipped = 0

    for record in records:
        messages = record["messages"]

        prompt_messages = messages[:-1]
        full_messages = messages

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Keep the end of the conversation if it exceeds the limit.
        if len(full_ids) > MAX_LENGTH:
            full_ids = full_ids[:MAX_LENGTH]

        prompt_length = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        labels[:prompt_length] = [-100] * prompt_length

        if all(label == -100 for label in labels):
            skipped += 1
            continue

        prepared.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        })

    print("Usable training examples:", len(prepared))
    print("Skipped examples:", skipped)

    if not prepared:
        raise ValueError("No usable training examples were prepared.")

    return Dataset.from_list(prepared)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch installation."
        )

    # Never overwrite an existing training run.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {OUTPUT_DIR}\n"
            "Existing work is protected. Use a new output folder."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):",
          round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

    records = load_dataset_records()

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("\nLoading base model in 4-bit mode...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
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

    print("\nPreparing dataset...")
    train_dataset = prepare_examples(records, tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("\nStarting KAIRO V4 training...")
    print("Base model:", BASE_MODEL)
    print("Dataset:", DATA_PATH)
    print("Output:", OUTPUT_DIR)
    print("Epochs: 2")
    print("Learning rate: 3e-5")

    result = trainer.train()

    print("\nTraining completed.")
    print("Training loss:", result.training_loss)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nKAIRO V4 adapter saved successfully:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
