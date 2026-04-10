import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

CKPT = os.environ.get(
    "CKPT",
    "/data/nfs-hpc/llm-express/checkpoints/gpt2-tinystories-express/checkpoint-2000"
)
BASE = os.environ.get("BASE_MODEL", "openai-community/gpt2")

prompts = [
    "Once upon a time,",
    "The little robot said,",
    "In a small village, there was",
]

gen_kwargs = dict(
    max_new_tokens=90,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
)

def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_id)
    return tok, mdl

def generate(tok, mdl, prompt, device):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(**inputs, **gen_kwargs)
    return tok.decode(out[0], skip_special_tokens=True)

def run_one(label, model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, mdl = load(model_id)
    mdl.to(device).eval()
    print("\n" + "#" * 80)
    print(f"{label}: {model_id}")
    print(f"device={device}")
    print("#" * 80)
    for i, p in enumerate(prompts, 1):
        set_seed(42 + i)
        text = generate(tok, mdl, p, device)
        print(f"\n=== {label} SAMPLE {i} ===")
        print(text)

def main():
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    run_one("BASE", BASE)
    run_one("CKPT", CKPT)

if __name__ == "__main__":
    main()
