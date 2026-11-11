
import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# --------------------------------------------------
# KAIRO V6 CONFIGURATION
# --------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = ROOT / "output_v6" / "kairo-1.5b-tanglish-v6"

MAX_NEW_TOKENS = 220

SYSTEM_PROMPT = """
You are KAIRO, a helpful coding and learning assistant.

For Tanglish questions:
- Answer in natural spoken Tamil written using English letters.
- Do not use Tamil script.
- Do not translate the question into English.
- Do not add unrelated code.
- Answer directly and simply.

For coding questions:
- Follow the exact user request.
- If asked for a complete program, include input handling.
- Do not hardcode example inputs when user input is requested.
- Return correct, runnable code.
"""

TESTS = [
    {
        "name": "Tanglish - Machine Learning",
        "question": (
            "Machine Learning na enna? Natural Tanglish-la "
            "exactly 4 simple sentences-la explain pannu. "
            "Tamil script use panna vendaam. Code vendaam."
        ),
        "kind": "tanglish",
    },
    {
        "name": "Tanglish - Cloud Computing",
        "question": (
            "Cloud computing na enna? Daily life example-oda "
            "simple natural Tanglish-la 4 sentences explain pannu. "
            "English letters mattum use pannu. Code vendaam."
        ),
        "kind": "tanglish",
    },
    {
        "name": "Tanglish - DBSCAN",
        "question": (
            "DBSCAN clustering na enna? Core point, border point, "
            "noise point-ai simple natural Tanglish-la explain pannu. "
            "Tamil script use panna vendaam. Code vendaam."
        ),
        "kind": "tanglish",
    },
    {
        "name": "Python - Even or Odd",
        "question": (
            "Write a complete Python program that reads an integer "
            "using input() and prints Even or Odd. Do not hardcode "
            "the number. Return code only."
        ),
        "kind": "even_odd",
    },
    {
        "name": "Python - Factorial",
        "question": (
            "Write a complete Python program that reads an integer "
            "using input(), rejects negative numbers, prints the "
            "factorial, and correctly handles 0! = 1. Return code only."
        ),
        "kind": "factorial",
    },
    {
        "name": "Python - Print 1 to 10",
        "question": (
            "Write Python code to print integers from 1 to 10, "
            "one number per line. Return code only."
        ),
        "kind": "print_1_to_10",
    },
    {
        "name": "Python - Square",
        "question": (
            "Write a complete Python program that reads a number "
            "using input() and prints its square. Do not hardcode "
            "the number. Return code only."
        ),
        "kind": "square",
    },
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():
    print("GPU:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "CPU")

    print("Base model:", BASE_MODEL)
    print("Loading KAIRO V6 adapter from:")
    print(ADAPTER_PATH)

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"V6 adapter folder not found:\n{ADAPTER_PATH}\n"
            "Check whether V6 training completed successfully."
        )

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    for filename in required_files:
        if not (ADAPTER_PATH / filename).exists():
            raise FileNotFoundError(
                f"Missing adapter file: {ADAPTER_PATH / filename}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH),
        local_files_only=True,
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

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            local_files_only=True,
        )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        is_trainable=False,
        local_files_only=True,
    )

    model.eval()

    print("\nKAIRO V6 loaded successfully.")
    print("Starting quality tests...\n")

    return model, tokenizer


# --------------------------------------------------
# GENERATE ANSWER
# --------------------------------------------------

def generate_answer(model, tokenizer, question):
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
        truncation=True,
        max_length=1024,
    )

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer


# --------------------------------------------------
# EXTRACT PYTHON CODE
# --------------------------------------------------

def extract_code(answer):
    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return answer.strip()


# --------------------------------------------------
# CHECK WHETHER CODE USES INPUT
# --------------------------------------------------

def uses_input(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "input":
                return True
    return False


def has_hardcoded_example_call(tree, function_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == function_name
                and node.args
                and all(
                    isinstance(arg, (ast.Constant, ast.Num))
                    for arg in node.args
                )
            ):
                return True
    return False


# --------------------------------------------------
# RUN GENERATED CODE IN A SEPARATE PROCESS
# --------------------------------------------------

def execute_code(code, test_input="", timeout=4):
    """
    Run the generated code in a separate Python process.
    This is not a security sandbox; only use it for code
    generated by this local model during your own testing.
    """

    with tempfile.TemporaryDirectory(prefix="kairo_v6_test_") as temp_dir:
        script_path = Path(temp_dir) / "generated_test.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=temp_dir,
            )

            return {
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }

        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Execution timed out.",
            }


# --------------------------------------------------
# CHECK CODING ANSWERS
# --------------------------------------------------

def check_python_code(answer, kind):
    code = extract_code(answer)

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return (
            "FAIL: Python syntax error: "
            f"{error.msg} at line {error.lineno}"
        )

    if kind == "even_odd":
        if not uses_input(tree):
            return "FAIL: Program does not read user input."

        if has_hardcoded_example_call(tree, "even_or_odd"):
            return "FAIL: Hardcoded function call detected."

        even_result = execute_code(code, "4\n")
        odd_result = execute_code(code, "5\n")

        if even_result["returncode"] != 0:
            return "FAIL: Even input execution failed: " + even_result["stderr"]

        if odd_result["returncode"] != 0:
            return "FAIL: Odd input execution failed: " + odd_result["stderr"]

        if "even" not in even_result["stdout"].lower():
            return "FAIL: Input 4 did not produce Even."

        if "odd" not in odd_result["stdout"].lower():
            return "FAIL: Input 5 did not produce Odd."

        return "PASS: Input 4 -> Even; input 5 -> Odd."

    if kind == "factorial":
        if not uses_input(tree):
            return "FAIL: Program does not read user input."

        zero_result = execute_code(code, "0\n")
        five_result = execute_code(code, "5\n")
        negative_result = execute_code(code, "-2\n")

        if zero_result["returncode"] != 0:
            return "FAIL: Zero input execution failed."

        if five_result["returncode"] != 0:
            return "FAIL: Input 5 execution failed."

        if negative_result["returncode"] != 0:
            return "FAIL: Negative input execution failed."

        if not re.search(r"\b1\b", zero_result["stdout"]):
            return "FAIL: Factorial of zero should be 1."

        if not re.search(r"\b120\b", five_result["stdout"]):
            return "FAIL: Factorial of 5 should be 120."

        if not any(
            word in negative_result["stdout"].lower()
            for word in ("invalid", "negative", "error")
        ):
            return "FAIL: Negative input is not clearly rejected."

        return "PASS: 0! = 1; 5! = 120; negative input rejected."

    if kind == "print_1_to_10":
        result = execute_code(code)

        if result["returncode"] != 0:
            return "FAIL: Execution error: " + result["stderr"]

        expected = [str(number) for number in range(1, 11)]
        actual = result["stdout"].splitlines()

        if actual != expected:
            return "FAIL: Output is not exactly numbers 1 to 10."

        return "PASS: Printed 1 through 10 correctly."

    if kind == "square":
        if not uses_input(tree):
            return "FAIL: Program does not read user input."

        result = execute_code(code, "6\n")

        if result["returncode"] != 0:
            return "FAIL: Execution error: " + result["stderr"]

        if not re.search(r"\b36(?:\.0)?\b", result["stdout"]):
            return "FAIL: Input 6 did not produce square 36."

        return "PASS: Input 6 -> square 36."

    return "MANUAL REVIEW: Check language and correctness."


# --------------------------------------------------
# CHECK TANGlish OUTPUT
# --------------------------------------------------

def check_tanglish(answer):
    if not answer.strip():
        return "FAIL: Empty answer."

    # Tamil Unicode characters are in these ranges.
    tamil_script = re.search(r"[\u0B80-\u0BFF]", answer)

    # Check whether the model returned a code block.
    code_block = "```" in answer

    if tamil_script:
        return "FAIL: Tamil script detected; Latin letters requested."

    if code_block:
        return "FAIL: Unexpected code block in explanation."

    if len(answer.split()) < 12:
        return "MANUAL REVIEW: Answer may be too short."

    # A language check cannot reliably judge natural Tanglish
    # using a simple automated rule.
    return "MANUAL REVIEW: Check natural Tanglish and factual accuracy."


# --------------------------------------------------
# RUN ALL TESTS
# --------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA GPU is not available.")
        print("CPU loading may be slow.")

    model, tokenizer = load_model()
    results = []

    for test in TESTS:
        print("=" * 70)
        print("TEST:", test["name"])
        print("QUESTION:", test["question"])
        print()

        try:
            answer = generate_answer(
                model,
                tokenizer,
                test["question"],
            )
        except Exception as error:
            result = f"ERROR: {type(error).__name__}: {error}"
            print(result)
            results.append((test["name"], result))
            continue

        print("ANSWER:\n")
        print(answer)
        print()

        if test["kind"] == "tanglish":
            result = check_tanglish(answer)
        else:
            result = check_python_code(answer, test["kind"])

        print("RESULT:", result)
        results.append((test["name"], result))
        print()

    print("=" * 70)
    print("KAIRO V6 QUALITY TEST SUMMARY")

    for name, result in results:
        print(f"{name}: {result}")

    print("\nTesting completed.")
    print("Adapter tested:", ADAPTER_PATH)
    print("No training or adapter files were modified.")


if __name__ == "__main__":
    main()
