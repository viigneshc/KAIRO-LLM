
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ==================================================
# KAIRO MODEL CONFIGURATION
# ==================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = r"output_v9/kairo-1.5b-tanglish-v9"

# Choose: "base" or "v9"
# base = test original model only
# v9   = test model with your existing V9 adapter
MODEL_MODE = "v9"
# ==================================================
# QUESTION TO TEST
# ==================================================

question = (
    "Machine learning na enna? "
    "Oru real-life example use panni simple Tanglish-la explain pannu."
)

# English test:
# question = "What is machine learning? Explain in simple English."

# Tamil script test:
# question = "ÃƒÂ Ã‚Â®Ã¢â‚¬Â¡ÃƒÂ Ã‚Â®Ã‚Â¯ÃƒÂ Ã‚Â®Ã‚Â¨ÃƒÂ Ã‚Â¯Ã‚ÂÃƒÂ Ã‚Â®Ã‚Â¤ÃƒÂ Ã‚Â®Ã‚Â¿ÃƒÂ Ã‚Â®Ã‚Â°ÃƒÂ Ã‚Â®Ã¢â‚¬Â¢ÃƒÂ Ã‚Â¯Ã‚Â ÃƒÂ Ã‚Â®Ã¢â‚¬Â¢ÃƒÂ Ã‚Â®Ã‚Â±ÃƒÂ Ã‚Â¯Ã‚ÂÃƒÂ Ã‚Â®Ã‚Â±ÃƒÂ Ã‚Â®Ã‚Â²ÃƒÂ Ã‚Â¯Ã‚Â ÃƒÂ Ã‚Â®Ã…Â½ÃƒÂ Ã‚Â®Ã‚Â©ÃƒÂ Ã‚Â¯Ã‚ÂÃƒÂ Ã‚Â®Ã‚Â±ÃƒÂ Ã‚Â®Ã‚Â¾ÃƒÂ Ã‚Â®Ã‚Â²ÃƒÂ Ã‚Â¯Ã‚Â ÃƒÂ Ã‚Â®Ã…Â½ÃƒÂ Ã‚Â®Ã‚Â©ÃƒÂ Ã‚Â¯Ã‚ÂÃƒÂ Ã‚Â®Ã‚Â©? ÃƒÂ Ã‚Â®Ã…Â½ÃƒÂ Ã‚Â®Ã‚Â³ÃƒÂ Ã‚Â®Ã‚Â¿ÃƒÂ Ã‚Â®Ã‚Â®ÃƒÂ Ã‚Â¯Ã‹â€ ÃƒÂ Ã‚Â®Ã‚Â¯ÃƒÂ Ã‚Â®Ã‚Â¾ÃƒÂ Ã‚Â®Ã¢â‚¬Â¢ ÃƒÂ Ã‚Â®Ã‚ÂµÃƒÂ Ã‚Â®Ã‚Â¿ÃƒÂ Ã‚Â®Ã‚Â³ÃƒÂ Ã‚Â®Ã¢â‚¬Â¢ÃƒÂ Ã‚Â¯Ã‚ÂÃƒÂ Ã‚Â®Ã¢â‚¬Â¢ÃƒÂ Ã‚Â®Ã‚ÂµÃƒÂ Ã‚Â¯Ã‚ÂÃƒÂ Ã‚Â®Ã‚Â®ÃƒÂ Ã‚Â¯Ã‚Â."

# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are KAIRO, a helpful AI assistant.

Language instructions:
- English request: answer in English.
- Tanglish request: answer in Tamil written using English letters.
- Tamil script request: answer in Tamil script.
- Follow the requested language and format.

Answer instructions:
- Answer the question directly.
- Explain the concept in simple words.
- Use one relevant real-life example when useful.
- Avoid repeating words, phrases, or sentences.
- Do not output unrelated technical terms.
- Do not repeat the user's question.
- Be honest when you do not know something.

For Tanglish, use natural conversational Tamil written
in English letters. Do not switch to Tamil script unless
the user requests Tamil script.
"""

# ==================================================
# LOAD MODEL
# ==================================================

def load_model():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Check your PyTorch installation."
        )

    print("Loading base model in 4-bit mode...")

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
    )

    if MODEL_MODE.lower() == "v9":
        print("Loading existing V9 adapter...")

        model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH,
            is_trainable=False,
        )

    elif MODEL_MODE.lower() != "base":
        raise ValueError(
            'MODEL_MODE must be either "base" or "v9".'
        )

    model.eval()

    # Clean generation settings for deterministic testing
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return tokenizer, model


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(tokenizer, model, question):
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

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda:0")

    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.10,
            no_repeat_ngram_size=4,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[-1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    del input_ids, attention_mask, output_ids, new_tokens

    return answer


# ==================================================
# MAIN
# ==================================================

def main():
    print("=" * 55)
    print("KAIRO MODEL TEST")
    print("Mode:", MODEL_MODE)
    print("=" * 55)

    tokenizer, model = load_model()

    print("\nType your question below.")
    print("Type exit to stop testing.")

    try:
        while True:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            print("\nGenerating answer...\n")

            answer = generate_answer(
                tokenizer,
                model,
                question,
            )

            print("=" * 55)
            print("KAIRO ANSWER")
            print("=" * 55)
            print(answer if answer else "[No answer generated]")
            print("=" * 55)

    except (KeyboardInterrupt, EOFError):
        print("\nTest stopped.")

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()






