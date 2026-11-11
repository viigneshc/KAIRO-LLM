
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ==========================================
# KAIRO - Local AI Assistant
# Model: Qwen2.5-1.5B-Instruct
# GPU: NVIDIA RTX 2050 (4GB VRAM)
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """
You are KAIRO, a helpful and accurate AI assistant.

LANGUAGE RULES:
1. Always follow the language requested by the user.
2. If the user asks for Tanglish, answer ONLY in natural Tanglish.
3. Tanglish means spoken Tamil written using English letters.
4. Never use Hindi unless the user explicitly requests it.
5. For English requests, answer in clear, simple English.
6. Use simple explanations and practical examples.

TANGLISH EXAMPLE:
User: Cloud computing na enna? Tanglish la explain pannu.
Assistant: Cloud computing na internet moolama computer resources
use panradhu. Example-ku, Google Drive la files save pannina,
namma computer-la mattum illaama internet moolama access pannalam.

ACCURACY RULES:
1. Give accurate answers. If unsure, clearly say so.
2. For calculations, verify the answer before responding.
3. For programming requests, provide complete runnable code.
4. Explain important parts of code briefly.
5. Do not invent facts.
6. Avoid repeating the same sentence unnecessarily.
7. Keep the answer relevant to the user's question.
"""

def main():
    print("=" * 55)
    print("              KAIRO LOCAL AI")
    print("=" * 55)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is not available.")
        print("Check your PyTorch and NVIDIA driver installation.")
        return

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Loading tokenizer...")

    # Load model files from local Hugging Face cache
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    # 4-bit quantization to reduce GPU memory usage
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading Qwen 1.5B model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    # Deterministic generation settings; remove conflicting defaults
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.use_cache = True

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"Model loaded successfully. GPU memory: {allocated:.0f} MB")
    print("=" * 55)
    print("Type 'exit' to close KAIRO.")
    print("Type 'clear' to clear conversation history.")
    print("=" * 55)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKAIRO closed.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "bye"):
            print("KAIRO: Goodbye!")
            break

        if question.lower() == "clear":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            print("KAIRO: Conversation cleared.")
            continue

        messages.append({
            "role": "user",
            "content": question,
        })

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            inputs = {
                key: value.to("cuda:0")
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            # Decode only newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            new_tokens = output[0][input_length:]

            answer = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            print("\nKAIRO:", answer if answer else
                  "Ennaala answer generate panna mudiyala. Try again.")

            messages.append({
                "role": "assistant",
                "content": answer,
            })

            # Keep recent conversation from growing indefinitely
            if len(messages) > 9:
                messages = [
                    messages[0]
                ] + messages[-8:]

        except torch.cuda.OutOfMemoryError:
            print("\nERROR: GPU memory full.")
            print("Shorter question try pannu; vera GPU apps close pannu.")
            torch.cuda.empty_cache()

            # Remove the unanswered user message
            if messages and messages[-1]["role"] == "user":
                messages.pop()

        except Exception as error:
            print("\nERROR:", str(error))
            print("Check the error message and try again.")

            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    main()
