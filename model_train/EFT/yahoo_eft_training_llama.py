#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved SFT training script optimized for the Yahoo QA topic classification task.
Fixes data formatting and training configuration issues.
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
    """Formatting function used by the new SFTTrainer version."""
    return examples["text"]


def load_and_format_dataset(dataset_path, task_instruction, skip_failed=True):
    """Load and format the Yahoo QA topic classification dataset.

    Yahoo data format (outer example):
    {
        "instruction": "...",
        "input": "Requirements: ...",
        "output": "{\"input\": \"Text: Question: ...\nAnswer: ...\", \"output\": \"Education & Reference\"}"
    }

    We parse the inner JSON string to extract the QA text and topic label.
    The unified formatted template is:

    Instruction: <task_instruction>\n\nText:\n<qa_content>\n\nResponse: <topic_label>

    Returns: HF Dataset["text"]
    """
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    total = len(raw_data)
    formatted = []
    failed = 0
    allowed_labels = {
        "Society & Culture", "Science & Mathematics", "Health", 
        "Education & Reference", "Computers & Internet", "Sports",
        "Business & Finance", "Entertainment & Music", 
        "Family & Relationships", "Politics & Government"
    }

    for idx, item in enumerate(raw_data):
        try:
            nested_raw = item.get('output')
            if not isinstance(nested_raw, str):
                raise ValueError("outer 'output' not str")
            nested = json.loads(nested_raw)
            qa_content = nested.get('input', '') or ''
            topic_label = nested.get('output', '') or ''

            # Normalize QA content prefix "Text: "
            if qa_content.startswith('Text: '):
                qa_content = qa_content[6:].strip()

            topic_label = topic_label.strip()
            if topic_label not in allowed_labels:
                raise ValueError(f"topic label '{topic_label}' not in allowed set {allowed_labels}")

            full_text = (
                f"Instruction: {task_instruction}\n\n"
                f"Text:\n{qa_content}\n\n"
                f"Response: {topic_label}"
            )
            formatted.append({"text": full_text})
        except Exception:
            failed += 1
            if not skip_failed:
                import traceback
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
    """Train the Yahoo topic classification model with SFT and LoRA."""
    print("Starting improved SFT training...")

    # LoRA config
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

    # Split dataset
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
        formatting_func=formatting_func,  # Use formatting_func instead of dataset_text_field
    )

    # Start training
    trainer.train()

    # Save model
    trainer.save_model()
    print(f"Model saved to {config['output_dir']}")

    return trainer

def main():
    # Configuration
    config = {
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'dataset_path': './yahoo_train_1500.json',
        'output_dir': './merged_sft_grpo_sft_yahoo_model',
        'max_seq_length': 1024,  # Yahoo QA content is short; 1024 is enough
        'num_train_epochs': 5,
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': 8,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'logging_steps': 25,
        'save_steps': 200,
        'eval_steps': 200,
    }
    
    print("=== Yahoo QA Topic Classification SFT Training ===")
    print(f"Dataset: {config['dataset_path']} (Yahoo QA topic classification)")
    print(f"Model: {config['model_name']}")
    print(f"Output: {config['output_dir']}")
    print("Task: Topic classification (10-way)")

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load dataset (parse nested JSON)
    task_instruction = 'Classify the given Yahoo QA content into one of the following topic categories: Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government. Respond with only the exact topic label.'
    dataset = load_and_format_dataset(config['dataset_path'], task_instruction)

    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config['model_name'])

    # Train model
    trainer = train_model(dataset, model, tokenizer, config)

    print("Yahoo QA topic classification training completed successfully!")
    print(f"Model should now classify Yahoo QA content into topic categories")

if __name__ == "__main__":
    main()
