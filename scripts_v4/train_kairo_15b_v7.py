
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
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data_v7" / "kairo_bilingual_v7_clean.jsonl"
OUTPUT_PATH = ROOT / "output_v7" / "kairo-1.5b-tanglish-v7"

MAX_LENGTH = 512
EPOCHS = 2
LEARNING_RATE = 3e-5


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU detect aagala. Training start pannala."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}"
        )

    # Existing model files-ai overwrite panna koodadhu.
    if OUTPUT_PATH.exists() and any(OUTPUT_PATH.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files:\n{OUTPUT_PATH}\n"
            "Existing files safe-a irukkum. Pudhu output path use pannunga."
        )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATA_PATH)
    print("Output:", OUTPUT_PATH)

    records = []
    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            messages = item.get("messages", [])

            if not any(m.get("role") == "user" for m in messages):
                raise ValueError(f"Missing user message at line {line_number}")

            if not any(m.get("role") == "assistant" for m in messages):
                raise ValueError(
                    f"Missing assistant message at line {line_number}"
                )

            records.append(item)

    if not records:
        raise ValueError("Dataset empty-a irukku.")

    print("Training records:", len(records))

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_record(record):
        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(records).map(
        format_record,
        remove_columns=Dataset.from_list(records).column_names,
    )

    def tokenize_record(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_record,
        remove_columns=["text"],
    )

    # QLoRA 4-bit setup for a low-VRAM GPU.
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

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    print("\nKAIRO V7 training start aagudhu...")
    trainer.train()

    model.save_pretrained(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    print("\nV7 training complete!")
    print("Adapter saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
