
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
    "You are a helpful assistant. For Tanglish requests, "
    "use natural Tamil written only in English letters. "
    "Never use Tamil script in Tanglish answers. "
    "Follow the requested language and length."
)

QUESTIONS = [
    "Machine learning na enna? Simple-ah 3 sentences-la sollu.",
    "Cloud computing na enna? 2 sentences-la explain pannu.",
    "DBSCAN na enna? Tanglish-la short-ah explain pannu.",
]

def generate(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU detect aagala.")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Base model load aagudhu...")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    base.eval()
    base.generation_config.temperature = None
    base.generation_config.top_p = None
    base.generation_config.top_k = None

    print("V8 adapter attach aagudhu...")

    v8 = PeftModel.from_pretrained(
        base,
        str(ADAPTER_PATH),
        local_files_only=True,
    )
    v8.eval()

    for question in QUESTIONS:
        print("\n" + "=" * 60)
        print("QUESTION:", question)

        print("\nBASE MODEL:")
        print(generate(v8.get_base_model(), tokenizer, question))

        print("\nKAIRO V8:")
        print(generate(v8, tokenizer, question))

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
