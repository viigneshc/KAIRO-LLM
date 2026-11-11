
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = ROOT / "output_v8" / "kairo-1.5b-tanglish-v8"

SYSTEM_PROMPT = (
    "You are KAIRO, a helpful and accurate AI assistant. "
    "Answer English questions in clear English. For Tanglish requests, "
    "use natural Tamil written only in English letters. Never use Tamil "
    "script in Tanglish answers. Follow the requested language, length, "
    "and format. Do not add code unless requested. Answer directly and "
    "avoid repetition. Never invent facts."
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU detect aagala.")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(f"Adapter missing: {ADAPTER_PATH}")

    print("GPU:", torch.cuda.get_device_name(0))
    print("KAIRO V8 load aagudhu...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH),
        local_files_only=True,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        local_files_only=True,
    )

    model.eval()

    # Greedy decoding use pannumbodhu sampling settings warnings avoid panna
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    print("\nKAIRO V8 ready!")
    print("'exit' type pannina program stop aagum.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKAIRO stopped.")
            break

        if question.lower() in {"exit", "quit"}:
            print("KAIRO stopped.")
            break

        if not question:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    eos_token_id=tokenizer.eos_token_id,
                )

            answer_tokens = outputs[0][
                inputs["input_ids"].shape[1]:
            ]

            answer = tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True,
            )

            print("\nKAIRO:", answer.strip())

        except torch.cuda.OutOfMemoryError:
            print("\nGPU memory podhavillai. Shorter question try pannu.")
            torch.cuda.empty_cache()

        except Exception as error:
            print("\nGeneration error:", error)


if __name__ == "__main__":
    main()
