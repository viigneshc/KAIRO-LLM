import ast
import io
import contextlib
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "output_v6/kairo-1.5b-tanglish-v6"

MAX_NEW_TOKENS = 350

TESTS = [
    {
        "name": "Tanglish - Machine Learning",
        "question": "Machine Learning na enna? Natural Tanglish la 4 sentences la explain pannu. Tamil words English letters la ezhuthu.",
        "type": "text",
    },
    {
        "name": "Tanglish - Cloud Computing",
        "question": "Cloud computing na enna? Pure natural Tanglish la oru simple real-life example oda explain pannu. Tamil script use panna vendam.",
        "type": "text",
    },
    {
        "name": "Tanglish - DBSCAN",
        "question": "DBSCAN clustering na enna? Core point, noise point concepts-ai simple Tanglish la explain pannu.",
        "type": "text",
    },
    {
        "name": "Python - Even or Odd",
        "question": "Write a complete Python program that reads an integer and prints Even or Odd. Return code only.",
        "type": "even_odd",
    },
    {
        "name": "Python - Factorial",
        "question": "Write a complete Python factorial function. Negative input must be rejected, 0 factorial must be 1. Return code only.",
        "type": "factorial",
    },
    {
        "name": "Python - Print 1 to 10",
        "question": "Write Python code to print numbers from 1 to 10. Return code only.",
        "type": "print_1_10",
    },
    {
        "name": "Python - Square",
        "question": "Write a Python function square(n) that returns n squared. Return code only.",
        "type": "square",
    },
]


def extract_code(answer):
    if "```" in answer:
        parts = answer.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("python"):
                cleaned = cleaned[6:].strip()
            if "def " in cleaned or "print(" in cleaned or "input(" in cleaned:
                return cleaned

    lines = answer.splitlines()
    code_lines = []
    started = False

    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("def ")
            or stripped.startswith("for ")
            or stripped.startswith("if ")
            or stripped.startswith("elif ")
            or stripped.startswith("else:")
            or stripped.startswith("print(")
            or stripped.startswith("n =")
            or stripped.startswith("number =")
            or stripped.startswith("result =")
            or stripped.startswith("return ")
            or stripped.startswith("try:")
            or stripped.startswith("except")
            or stripped.startswith("raise ")
            or stripped.startswith("while ")
            or stripped.startswith("input(")
        ):
            started = True

        if started:
            if stripped and not line.startswith((" ", "\t")):
                if (
                    stripped.startswith("This ")
                    or stripped.startswith("The ")
                    or stripped.startswith("For example")
                ):
                    break
            code_lines.append(line)

    return "\n".join(code_lines).strip()


def run_code_test(test_type, answer):
    code = extract_code(answer)

    if not code:
        return "FAIL: Python code not found"

    try:
        ast.parse(code)
    except SyntaxError as error:
        return f"FAIL: Syntax error: {error}"

    namespace = {}
    stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<kairo_answer>", "exec"), namespace)

            if test_type == "even_odd":
                if "check_even_odd" in namespace:
                    assert namespace["check_even_odd"](4) == "Even"
                    assert namespace["check_even_odd"](7) == "Odd"
                else:
                    # For input-based scripts, test using a controlled input.
                    import builtins

                    original_input = builtins.input
                    try:
                        builtins.input = lambda prompt="": "4"
                        stdout = io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            exec(compile(code, "<kairo_answer>", "exec"), {})
                    finally:
                        builtins.input = original_input

                    if "even" not in stdout.getvalue().lower():
                        return "FAIL: Even input did not print Even"

            elif test_type == "factorial":
                if "factorial" not in namespace:
                    return "FAIL: factorial function not found"

                fn = namespace["factorial"]
                assert fn(0) == 1, "0! must be 1"
                assert fn(1) == 1, "1! must be 1"
                assert fn(5) == 120, "5! must be 120"

                try:
                    result = fn(-3)
                    if not (
                        isinstance(result, str)
                        or isinstance(result, ValueError)
                    ):
                        return "FAIL: negative input was not rejected"
                except ValueError:
                    pass

            elif test_type == "print_1_10":
                output = stdout.getvalue()
                numbers = [
                    int(line.strip())
                    for line in output.splitlines()
                    if line.strip().isdigit()
                ]
                if numbers != list(range(1, 11)):
                    return f"FAIL: unexpected output: {output!r}"

            elif test_type == "square":
                if "square" not in namespace:
                    return "FAIL: square function not found"
                fn = namespace["square"]
                assert fn(4) == 16
                assert fn(-3) == 9
                assert fn(0) == 0

    except AssertionError as error:
        return f"FAIL: {error}"
    except Exception as error:
        return f"FAIL: Runtime error: {type(error).__name__}: {error}"

    return "PASS: Syntax and functional checks passed"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading KAIRO V6...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

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

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("\nKAIRO V6 loaded. Starting quality tests...\n")

    results = []

    for test in TESTS:
        print("=" * 70)
        print("TEST:", test["name"])
        print("QUESTION:", test["question"])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are KAIRO, a helpful AI assistant. "
                    "Follow the language requested by the user. "
                    "If Tanglish is requested, write Tamil in English "
                    "letters and use natural spoken Tamil-English mix. "
                    "Do not answer only in English when Tanglish is requested. "
                    "For coding, give correct, complete runnable Python code. "
                    "Follow the requested format."
                ),
            },
            {"role": "user", "content": test["question"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        print("\nANSWER:\n")
        print(answer)

        if test["type"] == "text":
            result = "MANUAL REVIEW: Check language and correctness"
        else:
            result = run_code_test(test["type"], answer)

        print("\nRESULT:", result)
        results.append((test["name"], result))

    print("\n" + "=" * 70)
    print("KAIRO V6 QUALITY TEST SUMMARY")

    for name, result in results:
        print(f"{name}: {result}")

    print("\nTesting completed. V5 adapter was not modified.")


if __name__ == "__main__":
    main()



