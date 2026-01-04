#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved SFT training script optimized for data synthesis.

Fixes dataset formatting and training configuration issues
for the Amazon sentiment task.
"""

import os
import json
import argparse
import traceback
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def formatting_func(examples):
    """Formatting function used by SFTTrainer.

    Input: a batch (dict) containing key 'text'.
    Output: list[str] of concatenated text samples.
    """
    return examples["text"]

def load_and_format_dataset(dataset_path, task_instruction, skip_failed=True):
    """Load and format the Amazon sentiment classification dataset.

    Amazon data format (outer example):
    {
        "output": "{\"input\": \"Text: ...\", \"output\": \"very positive\"}"
    }

    We parse the inner JSON string to extract the review text and label,
    and format them into the unified template:

    Instruction: <task_instruction>\n\nReview:\n<review>\n\nResponse: <label>

    Returns a HuggingFace Dataset with a single 'text' field.
    """
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    total = len(raw_data)
    formatted = []
    failed = 0
    allowed_labels = {"very negative", "negative", "neutral", "positive", "very positive"}

    for idx, item in enumerate(raw_data):
        try:
            nested_raw = item.get('output')
            if not isinstance(nested_raw, str):
                raise ValueError("outer 'output' not str")
            nested = json.loads(nested_raw)
            review = nested.get('input', '') or ''
            label = nested.get('output', '') or ''

            # Normalize review prefix "Text: " and variants
            prefix = 'text:'
            if review.lower().startswith(prefix):
                # Keep everything after the colon
                review = review[len(prefix):].strip()
            if review.lower().startswith('text:'):
                review = review[5:].strip()
            if review.startswith('Text: '):
                review = review[6:].strip()

            label = label.strip()
            if label not in allowed_labels:
                # Try a simple lowercase normalization first
                lower_label = label.lower()
                if lower_label in allowed_labels:
                    label = lower_label
                else:
                    raise ValueError(f"label '{label}' not in allowed set {allowed_labels}")

            full_text = (
                f"Instruction: {task_instruction}\n\n"
                f"Review:\n{review}\n\n"
                f"Response: {label}"
            )
            formatted.append({"text": full_text})
        except Exception:
            failed += 1
            if not skip_failed:
                traceback.print_exc()
            continue

    print(f"Loaded {total} raw samples; formatted {len(formatted)}; failed {failed}")
    if len(formatted) == 0:
        raise RuntimeError("No samples formatted successfully. Please check dataset format.")
    return Dataset.from_list(formatted)

def setup_model_and_tokenizer(model_name):
    """Set up model and tokenizer."""
    print(f"Loading model and tokenizer from {model_name}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora_config():
    """Set up LoRA configuration."""
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

def train_model(dataset, model, tokenizer, config):
    """Train the Amazon sentiment model with SFT and LoRA."""

    print("Starting improved SFT training...")

    # LoRA configuration
    peft_config = setup_lora_config()
    
    # SFT training configuration
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
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=False,
        report_to="none",
        max_seq_length=config['max_seq_length'],
        packing=False,
        save_total_limit=3,
        logging_dir=f"{config['output_dir']}/logs",
        lr_scheduler_type="linear",
        save_only_model=True,
    )
    
    # Split dataset into train/eval
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
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )
    
    # Create trainer
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
    
    # Start training
    trainer.train()

    # Save final model
    trainer.save_model()
    print(f"Model saved to {config['output_dir']}")

    return trainer

def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training for Amazon Sentiment Dataset with LoRA (Improved)")
    parser.add_argument('--model_name', type=str,
                        default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset_path', type=str,
                        default='./amazon_train_4000.json')
    parser.add_argument('--output_dir', type=str, default='./synthesis_model_output_amazon')
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=25)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--max_seq_length', type=int, default=1024,
                        help='Amazon reviews are relatively short; 1024 is usually enough.')
    parser.add_argument('--instruction', type=str, default='Classify the sentiment of the given Amazon product review into one of: very negative, negative, neutral, positive, very positive. Respond with only the label.')
    parser.add_argument('--no_bf16', action='store_true',
                        help='Disable bf16 (if the hardware does not support it).')
    return parser.parse_args()

def main():
    args = parse_args()

    config = {
        'model_name': args.model_name,
        'dataset_path': args.dataset_path,
        'output_dir': args.output_dir,
        'max_seq_length': args.max_seq_length,
        'num_train_epochs': args.num_train_epochs,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'per_device_eval_batch_size': args.per_device_eval_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'logging_steps': args.logging_steps,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
    }

    print("=== Amazon Sentiment SFT Training (Improved) ===")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Model: {config['model_name']}")
    print(f"Output: {config['output_dir']}")
    print("Task: Sentiment classification (5-way)")

    os.makedirs(config['output_dir'], exist_ok=True)

    # Load dataset (parse nested JSON)
    dataset = load_and_format_dataset(config['dataset_path'], args.instruction)

    # Set up model & tokenizer
    model, tokenizer = setup_model_and_tokenizer(config['model_name'])

    # Optionally disable bf16
    if args.no_bf16:
        print("bf16 disabled via --no_bf16 (currently relies on SFTConfig bf16=True; modify if needed).")

    trainer = train_model(dataset, model, tokenizer, config)

    print("Training completed successfully for Amazon sentiment task!")
    print(f"Adapter & outputs saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()
