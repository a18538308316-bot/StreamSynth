#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNLI SFT training script optimized for natural language inference.

Adapted from the Yelp training code to fit the MNLI
three-way classification task (entailment/contradiction/neutral).
"""

import os
import json
import argparse
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
    """Formatting function used by the newer SFTTrainer APIs."""
    return examples["text"]

def load_and_format_dataset(dataset_path):
    """Load and format the MNLI dataset into plain text."""
    print(f"Loading MNLI dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} MNLI samples")

    # Convert MNLI data into plain-text format for NLI tasks
    formatted_data = []
    for item in data:
        instruction = item.get('instruction', '')
        user_input = item.get('input', '')
        output = item.get('output', '')
        
        # Build a full training text in the following format:
        # <instruction>\n\n<input>\n\nResponse: <output><eos>
        # Specifically tailored for MNLI premise-hypothesis NLI task
        full_text = f"{instruction}\n\n{user_input}\n\nResponse: {output}"
        
        formatted_data.append({"text": full_text})
    
    print(f"Formatted {len(formatted_data)} MNLI samples for NLI training")
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer(model_name):
    """Set up model and tokenizer."""
    print(f"Loading model and tokenizer from {model_name}")

    # 4-bit quantization configuration
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
    """Train MNLI model."""
    print("Starting MNLI SFT training for Natural Language Inference...")

    # LoRA configuration
    peft_config = setup_lora_config()
    
    # SFT training configuration tuned for the MNLI task
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
    print("Task: MNLI three-way classification (entailment/contradiction/neutral)")

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
    print(f"MNLI model saved to {config['output_dir']}")

    return trainer

def main():
    # MNLI training configuration
    config = {
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'dataset_path': './MNLI_train_1496.json',
        'output_dir': './mnli_sft_grpo_sft_model_output',
        'max_seq_length': 2048,
        'num_train_epochs': 3,
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
    
    print("=== MNLI Natural Language Inference SFT Training ===")
    print(f"Dataset: {config['dataset_path']} (1496 training samples)")
    print(f"Model: {config['model_name']}")
    print(f"Output: {config['output_dir']}")
    print("Training objective: Learn to perform natural language inference (entailment/contradiction/neutral)")
    print("Task type: Three-way classification with premise-hypothesis pairs")

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load dataset
    dataset = load_and_format_dataset(config['dataset_path'])

    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config['model_name'])

    # Train model
    trainer = train_model(dataset, model, tokenizer, config)
    
    print("MNLI natural language inference training completed successfully!")
    print(f"Model should now perform better on NLI tasks with entailment/contradiction/neutral classification")

if __name__ == "__main__":
    main()
