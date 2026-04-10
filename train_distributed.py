#!/usr/bin/env python3
"""
train_distributed.py
Fine-tuning distribuido de GPT-2 sobre TinyStories con PyTorch DDP.
Funciona en 1 nodo (prueba local) o N nodos via torchrun + Slurm.
"""
import os
import time
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)


def group_texts(examples, block_size):
    concatenated = {k: sum(examples[k], []) for k in examples}
    total = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="openai-community/gpt2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    # torchrun inyecta estas variables antes de ejecutar el script
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    is_main = (global_rank == 0)  # solo el proceso 0 imprime y guarda

    if is_main:
        print("=== ENV CHECK ===")
        print(f"world_size: {world_size}  global_rank: {global_rank}  "
              f"local_rank: {local_rank}")
        print(f"torch: {torch.__version__}  CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print("=================")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if is_main:
        print("[1/4] Cargando dataset TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=args.cache_dir)

    if is_main:
        print("[2/4] Cargando tokenizer y modelo...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e9)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    if is_main:
        print("[3/4] Tokenizando y agrupando en bloques...")
    tokenized = ds.map(
        lambda b: tokenizer(b["text"], add_special_tokens=True),
        batched=True,
        num_proc=1,
        remove_columns=ds["train"].column_names,
        desc="Tokenizando",
    )
    lm_ds = tokenized.map(
        lambda x: group_texts(x, args.block_size),
        batched=True,
        num_proc=1,
        desc=f"Agrupando en bloques de {args.block_size}",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        report_to=[],
        seed=args.seed,
        fp16=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
    )

    if is_main:
        print("[4/4] Iniciando entrenamiento distribuido...")
    t0 = time.time()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        data_collator=collator,
    )
    trainer.train()

    if is_main:
        print(f"=== ENTRENAMIENTO TERMINADO: {time.time() - t0:.1f}s ===")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Modelo guardado en:", args.output_dir)


if __name__ == "__main__":
    main()
