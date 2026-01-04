#!/usr/bin/env python3
# -*- coding: utf-8 -*-
DEFAULT_CONFIG = {
    'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'max_seq_length': 2048,  # Reduce sequence length to avoid OOM
    'num_train_epochs': 2,   # Fewer epochs to avoid overfitting
    'per_device_train_batch_size': 2,  # Further reduce batch size
    'per_device_eval_batch_size': 2,
    'gradient_accumulation_steps': 8,  # More accumulation steps to keep effective batch size
    'learning_rate': 5e-5,   # Slightly higher learning rate
    'weight_decay': 0.01,
    'warmup_steps': 100,     # Fewer warmup steps
    'logging_steps': 25,
    'save_steps': 200,       # Save more frequently
    'eval_steps': 200,
    'output_dir': './synthesis_model_training_output'
}

def load_and_prepare_dataset(dataset_path, text_key='input', label_key='output'):
    """Load and prepare the dataset for synthesis training.

    Supports:
    - Chat-style data with "messages" field.
    - Triplet format with (instruction, input, output).
    - Simple (input, output) pairs.
    """
    print(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")

    # Check data format
    if len(data) > 0 and 'messages' in data[0]:
        # Chat format data, return directly
        print("Detected chat format data")
        return Dataset.from_list(data)
    
    # Check whether it is a three-field format (instruction, input, output)
    if len(data) > 0 and 'instruction' in data[0] and 'input' in data[0] and 'output' in data[0]:
        print("Detected synthesis training format (instruction, input, output)")
        formatted_data = []
        for item in data:
            instruction = item.get('instruction', '')
            user_input = item.get('input', '')
            output = item.get('output', '')

            # Merge instruction and input into system prompt + user input
            # This helps the model learn to generate synthetic data based on task description and requirements
            system_message = f"{instruction}\n\n{user_input}"
            
            conversation = [
                {"role": "user", "content": system_message},
                {"role": "assistant", "content": output}
            ]
            formatted_data.append({"messages": conversation})
        
        print(f"Formatted {len(formatted_data)} samples for synthesis training")
        return Dataset.from_list(formatted_data)

    # Handle simple input/output format for synthesis training
    formatted_data = []
    for item in data:
        text = item.get(text_key, '')
        label = item.get(label_key, '')

        # Create chat-style data for synthesis training
        # User input: contains original data and synthesis task description
        # Assistant output: high-quality synthetic data
        conversation = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": label}
        ]
        formatted_data.append({"messages": conversation})
    
    print(f"Formatted {len(formatted_data)} samples for synthesis training")
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer(model_name):
    """Set up model and tokenizer with 4-bit quantization."""
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
    """Set LoRA configuration optimized for data synthesis."""
    return LoraConfig(
        lora_alpha=16,  # Lower alpha to reduce overfitting
        lora_dropout=0.1,
        r=64,   # Lower rank to reduce parameter count
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

def train_model(dataset, model, tokenizer, config):
    """Train the model using SFT for data synthesis."""
    print("Starting SFT training...")

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
        # Data synthesis specific settings
        save_total_limit=3,  # Keep only the latest 3 checkpoints
        evaluation_strategy="steps",
        logging_dir=f"{config['output_dir']}/logs",
        lr_scheduler_type="linear",  # Linear LR schedule
        save_only_model=True,
    )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    dataset_size = len(dataset)
    eval_size = min(200, int(dataset_size * 0.05))  # 5% eval, max 200 samples

    indices = list(range(dataset_size))
    import random
    random.seed(42)  # Ensure reproducibility
    random.shuffle(indices)

    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:]

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    from transformers import EarlyStoppingCallback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=peft_config,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    trainer.save_model()
    print(f"Model saved to {config['output_dir']}")

    return trainer

def main():
    parser = argparse.ArgumentParser(description='Standard SFT Training')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--model_name', type=str, default=DEFAULT_CONFIG['model_name'], help='Base model name/path')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'], help='Output directory')
    parser.add_argument('--max_seq_length', type=int, default=DEFAULT_CONFIG['max_seq_length'], help='Max sequence length')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_CONFIG['num_train_epochs'], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['per_device_train_batch_size'], help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'], help='Learning rate')
    parser.add_argument('--text_key', type=str, default='input', help='Key for text field in JSON')
    parser.add_argument('--label_key', type=str, default='output', help='Key for label field in JSON')
    
    args = parser.parse_args()
    
    # Merge CLI args into config
    config = DEFAULT_CONFIG.copy()
    config.update({
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'max_seq_length': args.max_seq_length,
        'num_train_epochs': args.num_epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    })

    print("=== Synthesis Model SFT Training ===")
    print(f"Model: {config['model_name']}")
    print(f"Output: {config['output_dir']}")
    print("Training objective: Learn to generate high-quality synthetic data")

    os.makedirs(config['output_dir'], exist_ok=True)

    dataset = load_and_prepare_dataset(args.dataset_path, args.text_key, args.label_key)
    model, tokenizer = setup_model_and_tokenizer(config['model_name'])
    train_model(dataset, model, tokenizer, config)

    print("Synthesis model training completed successfully!")
    print(f"Model can now generate high-quality synthetic data from original inputs")

if __name__ == "__main__":
    main()
