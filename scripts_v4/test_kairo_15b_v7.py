
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = ROOT / "output_v7" / "kairo-1.5b-tanglish-v7"

SYSTEM_PROMPT = (
    "You are KAIRO, a helpful and accurate AI assistant. "
    "Answer English questions in clear English. For Tanglish requests, "
    "use natural Tamil written only in English letters. Never use Tamil "
    "script in Tanglish answers. Answer directly and avoid repetition. "
    "For coding requests, provide complete runnable programs when requested."
)

TESTS = [
    (
        "Tanglish - Machine Learning",
        "Machine Learning na enna? Natural Tanglish-la exactly 4 "
        "simple sentences-la explain pannu. Tamil script vendaam. Code vendaam.",
        "tanglish",
    ),
    (
        "Tanglish - Cloud Computing",
        "Cloud computing na enna? Daily life example-oda 4 sentences "
        "natural Tanglish-la explain pannu. English letters mattum use pannu.",
        "tanglish",
    ),
    (
        "Tanglish - DBSCAN",
        "DBSCAN na enna? Core point, border point, noise point-ai "
        "simple Tanglish-la explain pannu. Tamil script vendaam.",
        "tanglish",
    ),
    (
        "Python - Even or Odd",
        "Write a complete Python program that reads an integer using "
        "input() and prints Even or Odd. Return code only.",
        "even_odd",
    ),
    (
        "Python - Factorial",
        "Write a complete Python program that reads an integer, rejects "
        "negative numbers, and prints factorial. Handle 0! = 1. Return code only.",
        "factorial",
    ),
    (
        "Python - Print 1 to 10",
        "Write Python code to print numbers 1 to 10, one per line. Return code only.",
        "print_1_to_10",
    ),
    (
        "Python - Square",
        "Write a complete Python program that reads a number using input() "
        "and prints its square. Return code only.",
        "square",
    ),
]


def load_model():
    print("GPU:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "CPU")
    print("Loading V7 adapter:", ADAPTER_PATH)

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(f"V7 adapter not found: {ADAPTER_PATH}")

    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ADAPTER_PATH / filename).exists():
            raise FileNotFoundError(f"Missing file: {ADAPTER_PATH / filename}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            local_files_only=True,
        )

    model = PeftModel.from_pretrained(
        base, str(ADAPTER_PATH), is_trainable=False, local_files_only=True
    )
    model.eval()
    print("V7 loaded successfully.\n")
    return model, tokenizer


def generate_answer(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_code(answer):
    match = re.search(
        r"```(?:python)?\s*(.*?)```", answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else answer.strip()


def execute_code(code, test_input=""):
    with tempfile.TemporaryDirectory(prefix="kairo_v7_test_") as folder:
        script = Path(folder) / "generated.py"
        script.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=4,
                cwd=folder,
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Execution timed out."


def check_code(answer, kind):
    code = extract_code(answer)

    if kind == "even_odd":
        for value, expected in (("4\n", "even"), ("5\n", "odd")):
            rc, out, err = execute_code(code, value)
            if rc != 0 or expected not in out.lower():
                return f"FAIL: input {value.strip()} -> {out or err}"
        return "PASS: Even/odd inputs tested."

    if kind == "factorial":
        for value, expected in (("0\n", "1"), ("5\n", "120")):
            rc, out, err = execute_code(code, value)
            if rc != 0 or not re.search(rf"\b{expected}\b", out):
                return f"FAIL: input {value.strip()} -> {out or err}"
        rc, out, err = execute_code(code, "-2\n")
        if rc != 0 or not any(
            word in out.lower() for word in ("invalid", "negative", "error")
        ):
            return "FAIL: Negative input not clearly rejected."
        return "PASS: 0!, 5!, and negative input tested."

    if kind == "print_1_to_10":
        rc, out, err = execute_code(code)
        expected = "\n".join(str(n) for n in range(1, 11))
        return "PASS: Printed 1 to 10." if rc == 0 and out == expected else (
            f"FAIL: {out or err}"
        )

    if kind == "square":
        rc, out, err = execute_code(code, "6\n")
        if rc == 0 and re.search(r"\b36(?:\.0)?\b", out):
            return "PASS: Input 6 -> square 36."
        return f"FAIL: {out or err}"

    return "MANUAL REVIEW: Check generated code."


def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; CPU testing may be slow.")

    model, tokenizer = load_model()
    results = []

    for name, question, kind in TESTS:
        print("=" * 65)
        print("TEST:", name)
        try:
            answer = generate_answer(model, tokenizer, question)
            print("\nANSWER:\n", answer, "\n")
            if kind == "tanglish":
                if re.search(r"[\u0B80-\u0BFF]", answer):
                    result = "FAIL: Tamil script detected."
                elif len(answer.split()) < 12:
                    result = "MANUAL REVIEW: Answer may be too short."
                else:
                    result = "MANUAL REVIEW: Check natural Tanglish and accuracy."
            else:
                result = check_code(answer, kind)
        except Exception as error:
            result = f"ERROR: {type(error).__name__}: {error}"

        print("RESULT:", result)
        results.append((name, result))

    print("\n" + "=" * 65)
    print("KAIRO V7 TEST SUMMARY")
    for name, result in results:
        print(f"{name}: {result}")
    print("\nAdapter tested:", ADAPTER_PATH)
    print("Testing complete. No training files were modified.")


if __name__ == "__main__":
    main()
