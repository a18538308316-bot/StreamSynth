#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Qwen SFT training script for the MNLI task.

Default base model is Qwen2.5-7B-Instruct.
"""

import os
import json
import traceback
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def formatting_func(examples):
    return examples['text']


def load_and_format_dataset(dataset_path):
    print(f"Loading MNLI dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} MNLI samples")
    formatted_data = []
    for item in data:
        try:
            instruction = item.get('instruction','')
            nested_raw = item.get('output','')
            if isinstance(nested_raw, str):
                nested = json.loads(nested_raw)
                content = nested.get('input','')
                label = nested.get('output','')
            else:
                content = item.get('input','')
                label = item.get('label','')
            # ensure content contains premise and hypothesis
            text = f"Instruction: {instruction}\n\nText:\n{content}\n\nResponse: {label}"
            formatted_data.append({'text': text})
        except Exception:
            traceback.print_exc()
            continue
    print(f"Formatted {len(formatted_data)} MNLI samples for NLI training")
    return Dataset.from_list(formatted_data)


def setup_model_and_tokenizer(model_name):
    print(f"Loading model and tokenizer from {model_name}")
    # 4-bit quantization config preserved from Llama improved script
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def setup_lora_config():
    return LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )


def train_model(dataset, model, tokenizer, config):
    print('Starting Qwen MNLI SFT training...')
    peft_config = setup_lora_config()
    sft_config = SFTConfig(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=False,
        report_to='none',
        max_seq_length=config['max_seq_length'],
        packing=False,
        save_total_limit=3,
        logging_dir=f"{config['output_dir']}/logs",
        lr_scheduler_type='cosine',
        save_only_model=True,
        max_grad_norm=1.0,
    )

    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.05))
    import random
    random.seed(42)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:]

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.005
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        callbacks=[early_stopping_callback],
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model()
    print(f"Model saved to {config['output_dir']}")
    return trainer


def main():
    config = {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
        'dataset_path': './MNLI_train_1496.json',
        'output_dir': './mnli_sft_grpo_sft_qwen_model_output',
        'max_seq_length': 2048,
        'num_train_epochs': 3,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'warmup_steps': 50,
        'logging_steps': 25,
        'save_steps': 200,
        'eval_steps': 200,
    }

    print('=== Qwen MNLI SFT Training (Improved) ===')
    os.makedirs(config['output_dir'], exist_ok=True)

    dataset = load_and_format_dataset(config['dataset_path'])
    model, tokenizer = setup_model_and_tokenizer(config['model_name'])
    train_model(dataset, model, tokenizer, config)
    print('Qwen MNLI improved training completed!')

if __name__ == '__main__':
    main()
