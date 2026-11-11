
import ast
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
ADAPTER = "output_v3/kairo-1.5b-coding-v3"

SYSTEM_PROMPT = """
You are KAIRO, a helpful and accurate AI assistant.

Language:
- Answer English questions in clear English.
- Answer Tanglish questions in natural spoken Tanglish using English letters.
- Explain concepts in simple sentences, not just keyword lists.
- Include an example when requested.

Coding:
- Give complete, runnable code when requested.
- Follow the exact requirements.
- For even/odd, use num % 2 == 0.
- Avoid unrelated conditions.
- Check code for syntax and logical errors.
"""


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch CUDA installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")

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

    print("Loading KAIRO v3 adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER,
    )

    model.eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    print("KAIRO v3 loaded successfully.\n")

    return tokenizer, model


def ask_kairo(question, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=180,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    answer_ids = output[0][inputs.shape[-1]:]

    return tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    ).strip()


def extract_python_code(answer):
    # First, try to extract a fenced code block.
    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return answer.strip()


def check_syntax(code):
    try:
        ast.parse(code)
        return True, "Valid Python syntax"

    except SyntaxError as error:
        return (
            False,
            f"Line {error.lineno}: {error.msg}",
        )


def test_even_odd(code):
    results = []

    # Basic AST allowlist: only simple input, integer conversion,
    # printing, assignments and conditional expressions/statements.
    # This is a safeguard, not a complete security sandbox.
    allowed_nodes = {
        ast.Module,
        ast.Assign,
        ast.If,
        ast.Expr,
        ast.Name,
        ast.Call,
        ast.Constant,
        ast.Compare,
        ast.BinOp,
        ast.Mod,
        ast.Eq,
        ast.NotEq,
        ast.UnaryOp,
        ast.USub,
        ast.UAdd,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Gt,
        ast.Lt,
        ast.GtE,
        ast.LtE,
        ast.Load,
        ast.Store,
    }

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                return [
                    (
                        "-",
                        "-",
                        f"Rejected unsupported syntax: {type(node).__name__}",
                        False,
                    )
                ]

            if isinstance(node, ast.Name):
                allowed_names = {
                    "num",
                    "number",
                    "n",
                    "input",
                    "int",
                    "print",
                }

                if node.id not in allowed_names:
                    return [
                        (
                            "-",
                            "-",
                            f"Rejected name: {node.id}",
                            False,
                        )
                    ]

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    return [
                        ("-", "-", "Rejected function call", False)
                    ]

                if node.func.id not in {"input", "int", "print"}:
                    return [
                        (
                            "-",
                            "-",
                            f"Rejected function: {node.func.id}",
                            False,
                        )
                    ]

    except SyntaxError as error:
        return [
            (
                "-",
                "-",
                f"Syntax error: {error.msg}",
                False,
            )
        ]

    # Test positive, zero and negative values.
    test_cases = [
        ("2", "Even"),
        ("3", "Odd"),
        ("0", "Even"),
        ("-3", "Odd"),
        ("5", "Odd"),
        ("-4", "Even"),
    ]

    with tempfile.TemporaryDirectory(
        prefix="kairo_benchmark_"
    ) as folder:
        script_path = Path(folder) / "generated_program.py"
        script_path.write_text(code, encoding="utf-8")

        for number, expected in test_cases:
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(script_path),
                    ],
                    input=number + "\n",
                    capture_output=True,
                    text=True,
                    timeout=3,
                    cwd=folder,
                )

                output = process.stdout.strip()

                # Accept output such as "Enter a number: Even"
                # as long as the final output word is Even or Odd.
                match = re.search(
                    r"\b(Even|Odd)\b\s*$",
                    output,
                    flags=re.IGNORECASE,
                )

                passed = (
                    process.returncode == 0
                    and match is not None
                    and match.group(1).lower()
                    == expected.lower()
                )

                results.append(
                    (number, expected, output, passed)
                )

            except subprocess.TimeoutExpired:
                results.append(
                    (number, expected, "Timed out", False)
                )

            except Exception as error:
                results.append(
                    (number, expected, str(error), False)
                )

    return results


def run_language_tests(tokenizer, model):
    tests = [
        (
            "English ML",
            "What is machine learning? Explain simply.",
        ),
        (
            "Tanglish ML",
            "Machine learning na enna? Simple-ah explain pannu.",
        ),
        (
            "English cloud",
            "Explain cloud computing with an example.",
        ),
        (
            "Tanglish cloud",
            "Cloud computing na enna? Example-oda explain pannu.",
        ),
    ]

    print("=" * 55)
    print("PART 1: LANGUAGE QUALITY")
    print("=" * 55)

    for title, question in tests:
        print(f"\n[{title}]")
        print("Question:", question)

        try:
            answer = ask_kairo(
                question,
                tokenizer,
                model,
            )
            print("Answer:", answer)

        except Exception as error:
            print("ERROR:", error)

        print("-" * 50)


def run_coding_tests(tokenizer, model):
    tests = [
        (
            "Tanglish even/odd",
            "Python-la user kitta number input vaangi "
            "even or odd nu print panra program kudu.",
        ),
        (
            "English even/odd",
            "Write a Python program that takes a number "
            "as input and checks whether it is even or odd.",
        ),
    ]

    print("\n" + "=" * 55)
    print("PART 2: CODING CORRECTNESS")
    print("=" * 55)

    for title, question in tests:
        print(f"\n[{title}]")
        print("Question:", question)

        try:
            answer = ask_kairo(
                question,
                tokenizer,
                model,
            )

            print("\nGenerated answer:")
            print(answer)

            code = extract_python_code(answer)
            valid, message = check_syntax(code)

            print("\nSyntax check:", message)

            if not valid:
                print("Code tests skipped.")
                continue

            results = test_even_odd(code)
            passed_count = sum(
                1 for result in results if result[3]
            )

            print("\nTest results:")

            for number, expected, actual, passed in results:
                status = "PASS" if passed else "FAIL"

                print(
                    f"Input={number} | "
                    f"Expected={expected} | "
                    f"Output={actual!r} | "
                    f"{status}"
                )

            print(
                f"\nTests passed: "
                f"{passed_count}/{len(results)}"
            )

        except Exception as error:
            print("ERROR:", error)

        print("-" * 50)


def main():
    print("=" * 55)
    print("KAIRO v3 BENCHMARK")
    print("Language quality + coding correctness")
    print("=" * 55)

    try:
        tokenizer, model = load_model()

        run_language_tests(tokenizer, model)
        run_coding_tests(tokenizer, model)

        print("\nBenchmark completed.")

    except Exception as error:
        print("\nBenchmark failed:")
        print(type(error).__name__ + ":", error)


if __name__ == "__main__":
    main()
