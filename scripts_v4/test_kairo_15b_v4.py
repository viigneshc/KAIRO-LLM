
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "output_v4/kairo-1.5b-tanglish-v4"

SYSTEM_PROMPT = """
You are KAIRO, a helpful and accurate AI assistant.

LANGUAGE:
- Answer English questions in clear English.
- Answer Tanglish questions in natural spoken Tanglish using English letters.
- Never use Tamil script for Tanglish.
- Explain concepts simply, naturally, and step by step when needed.
- Avoid keyword-only answers when an explanation is requested.
- Use simple examples to make concepts understandable.

CODING:
- Follow the exact task requested.
- Return complete, runnable code when asked.
- Use the requested programming language.
- Use input() when user input is requested.
- For even or odd, use num % 2 == 0.
- Print Even if true; otherwise print Odd.
- Do not add unrelated conditions.
- Check function names and indentation.
- Never use undefined functions.
- For factorial, handle negative inputs appropriately.
"""

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading KAIRO V4 adapter...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER)
    model.eval()
    model.config.use_cache = True

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return tokenizer, model


def generate_answer(question, tokenizer, model):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    answer_ids = outputs[0][inputs.shape[-1]:]

    return tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    ).strip()


def main():
    print("=" * 48)
    print("KAIRO V4 - Evaluation")
    print("=" * 48)

    tokenizer, model = load_model()

    tests = [
        "What is machine learning? Explain simply.",
        "Machine learning na enna? Simple-ah explain pannu.",
        "Explain cloud computing with an example.",
        "Cloud computing na enna? Example-oda explain pannu.",
        "Python-la user kitta number input vaangi even or odd nu print panra program kudu.",
        "Write a Python program that takes a number as input and checks whether it is even or odd.",
        "Python-la 1 irundhu 10 vara numbers print panna code kudu.",
        "Write a Python program to find the factorial of a number.",
        "DBSCAN clustering na enna? Simple-ah example-oda explain pannu.",
    ]

    for question in tests:
        print("\nQUESTION:", question)

        try:
            answer = generate_answer(question, tokenizer, model)
            print("KAIRO V4:", answer)
        except torch.cuda.OutOfMemoryError:
            print("GPU memory full. Stop testing.")
            return
        except Exception as error:
            print("ERROR:", error)
            return

        print("-" * 48)

    print("\nInteractive mode. Type 'exit' to stop.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKAIRO stopped.")
            break

        if question.lower() in ("exit", "quit"):
            print("KAIRO stopped.")
            break

        if not question:
            continue

        try:
            print(
                "\nKAIRO V4:",
                generate_answer(question, tokenizer, model),
            )
        except torch.cuda.OutOfMemoryError:
            print("GPU memory full. Restart and try a shorter question.")
            break
        except Exception as error:
            print("ERROR:", error)


if __name__ == "__main__":
    main()
