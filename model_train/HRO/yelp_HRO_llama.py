#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward enhanced GRPO training script for Yelp.

This version focuses on positive rewards: it relaxes
scoring thresholds based on production analysis and
raises the effective reward ceiling to encourage learning.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset

# Add script path (to be customized in your environment)
sys.path.append('/path/to/TRL-GRPO/scripts_9_18')

# Import modular components
from scripts_9_18.data_utils import create_optimized_dataset
from scripts_9_18.reward_functions import (
    initialize_reward_globals,
    set_training_visualizer,
)
from scripts_9_18.training_visualizer import initialize_visualizer
from scripts_9_18.batch_diversity_reward import initialize_batch_diversity_calculator

# Import reward calculator related classes
from scripts_9_18.attribute_handler import AttrPromptAttributeLoader, AttributeComplianceCalculator, ATTRPROMPT_CONFIG

# =============================================================================
# Reward enhancement configuration
# =============================================================================

DATA_FILE = "/path/to/train_data_4000.json"
BASE_MODEL_NAME = "/path/to/TRL-GRPO/merged_sft_model"
SFT_LORA_PATH = "/path/to/TRL-SFT/synthesis_model_output_improved-52711"
MERGED_MODEL_PATH = "./merged_sft_model"
OUTPUT_DIR = "./synthesis_grpo_reward_enhanced_new_9_28_output-sim500-epo2-bat4-gr4-gen4"

USE_MERGED_MODEL = True

READABILITY_MODEL_PATH = "/path/to/reasoning-model"
EMBEDDING_MODEL_PATH = "/path/to/all-MiniLM-L6-v2"

# Reward-enhanced training configuration
ENHANCED_CONFIG = {
    "max_train_samples": 500,
    "num_train_epochs": 2.0,        # Slightly increase training epochs
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,           # Increase candidate generations for more choices
    "max_completion_length": 800,
    "logging_steps": 5,             
    "save_steps": 40,
    "learning_rate": 3e-6,          # Slightly higher learning rate
    "warmup_steps": 20,             # More warmup steps
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 4,
}

# 🎯 Re-tuned reward weight configuration
# Adjust weight allocation based on production performance
ENHANCED_SAMPLE_REWARDS_CONFIG = {
    "sentiment_consistency_weight": 0.1,    # Sentiment weight (most important)
    "attribute_compliance_weight": 0.1,     # Attribute weight (important)
    "generation_quality_weight": 0.5,       # Generation quality weight (includes length and JSON)
    # Remove standalone length reward; fold it into quality
}

ENHANCED_BATCH_REWARDS_CONFIG = {
    "batch_diversity_weight": 0.3,          # Diversity weight
}

# Generation hyper-parameters
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.85,            # Slightly higher creativity
    "top_p": 0.92,                  # Slightly relaxed
    "top_k": 60,                    # More vocabulary choices
    "repetition_penalty": 1.12,     # Slightly lower repetition penalty
    "pad_token_id": None,
}

# =============================================================================
# Reward-enhanced wrapper functions
# =============================================================================

def create_enhanced_reward_functions():
    """Create enhanced reward functions with relaxed thresholds and unified upper bound 1.0."""
    from scripts_9_18.reward_functions import (
        reward_sentiment_consistency_batch,
        reward_attribute_compliance_batch,
        reward_generation_quality_batch,
        update_current_prompt_attributes
    )
    from scripts_9_18.batch_diversity_reward import reward_batch_diversity
    
    # Reward call counter
    reward_call_counter = 0
    current_training_step = 0
    
    def enhanced_sentiment_consistency(completions, **kwargs):
        """Enhanced sentiment consistency reward - upper bound 1.0."""
        nonlocal reward_call_counter, current_training_step
        
        reward_call_counter += 1
        current_training_step = reward_call_counter // 4  # There are 4 reward functions now
        
        update_current_prompt_attributes(current_training_step)
        kwargs['step'] = current_training_step
        
        base_rewards = reward_sentiment_consistency_batch(completions, **kwargs)
        
        if hasattr(base_rewards, 'tolist'):
            reward_values = base_rewards.tolist()
        else:
            reward_values = base_rewards
        
        # 🎯 Map [0, 0.5] to [0, 1.0] to unify upper bound
        enhanced_rewards = []
        for r in reward_values:
            enhanced_r = r * 2.0  # Scale by 2x so upper bound becomes 1.0
            enhanced_rewards.append(min(enhanced_r, 1.0))
        
        return enhanced_rewards
    
    def enhanced_attribute_compliance(completions, **kwargs):
        """Enhanced attribute compliance reward - relaxed standard, upper bound 1.0."""
        kwargs['step'] = current_training_step
        base_rewards = reward_attribute_compliance_batch(completions, **kwargs)
        
        if hasattr(base_rewards, 'tolist'):
            reward_values = base_rewards.tolist()
        else:
            reward_values = base_rewards
        
        # 🎯 Relax attribute scoring: give high score when keywords exist
        # Map [0, 0.25] to [0.2, 1.0] for a looser scale
        enhanced_rewards = []
        for r in reward_values:
            if r > 0:  # Any attribute match gets a relatively high score
                enhanced_r = 0.5 + (r * 2.0)  # Base 0.5 + up to 0.5
            else:
                enhanced_r = 0.0
            enhanced_rewards.append(min(enhanced_r, 1.0))
        
        return enhanced_rewards
    
    def enhanced_generation_quality(completions, **kwargs):
        """Enhanced generation quality reward - includes length and JSON format, upper bound 1.0."""
        kwargs['step'] = current_training_step
        base_rewards = reward_generation_quality_batch(completions, **kwargs)
        
        if hasattr(base_rewards, 'tolist'):
            reward_values = base_rewards.tolist()
        else:
            reward_values = base_rewards
        
        # 🎯 Simplify quality scoring: focus mainly on JSON correctness
        # Map [0, 0.3] to [0.1, 1.0]
        enhanced_rewards = []
        for r in reward_values:
            if r >= 0.15:  # When base quality is acceptable
                enhanced_r = 0.4 + (r * 2.0)  # Base 0.4 + amplified
            elif r > 0:    # When there is some base score
                enhanced_r = 0.2 + (r * 1.5)  # Base 0.2 + moderate amplification
            else:
                enhanced_r = 0.0
            enhanced_rewards.append(min(enhanced_r, 1.0))
        
        return enhanced_rewards
    
    def enhanced_batch_diversity(completions, **kwargs):
        """Enhanced batch diversity reward - much lower thresholds, upper bound 1.0."""
        kwargs['step'] = current_training_step
        base_rewards = reward_batch_diversity(completions, **kwargs)
        
        if hasattr(base_rewards, 'tolist'):
            reward_values = base_rewards.tolist()
        else:
            reward_values = base_rewards if isinstance(base_rewards, list) else [base_rewards]
        
        # 🎯 Greatly lower diversity thresholds to adapt to fixed prompts
        enhanced_rewards = []
        for r in reward_values:
            # Because prompts are fixed, lower diversity requirements
            if r >= 0.3:      # Previously needed 0.7, now 0.3 yields high score
                enhanced_r = 0.8 + (r * 0.2)  # Base 0.8
            elif r >= 0.2:    # Previously needed 0.5, now 0.2 yields medium score
                enhanced_r = 0.5 + (r * 1.0)
            elif r >= 0.1:    # Previously needed 0.3, now 0.1 yields base score
                enhanced_r = 0.3 + (r * 1.0)
            else:
                enhanced_r = 0.1  # Give a minimal diversity score
            enhanced_rewards.append(min(enhanced_r, 1.0))
        
        return enhanced_rewards
    
    # Set function names for reward hooks
    enhanced_sentiment_consistency.__name__ = "reward_sentiment_consistency"
    enhanced_attribute_compliance.__name__ = "reward_attribute_compliance"
    enhanced_generation_quality.__name__ = "reward_generation_quality"
    enhanced_batch_diversity.__name__ = "reward_batch_diversity"
    
    return [
        enhanced_sentiment_consistency,
        enhanced_attribute_compliance,
        enhanced_generation_quality,
        enhanced_batch_diversity,
    ]

# =============================================================================
# Data processing functions
# =============================================================================

def load_and_process_data(data_file, max_samples=None):
    """Load and preprocess training data for GRPO."""
    print(f"Loading synthesis data from {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} samples")
    
    # Prepare GRPO dataset
    grpo_data = []
    for item in data:
        full_prompt = item.get('instruction', '') + '\n\n' + item.get('input', '')
        
        input_text = item.get('input', '')
        target_sentiment = 'neutral'
        
        if 'Target sentiment for generation:' in input_text:
            sentiment_line = input_text.split('Target sentiment for generation:')[1].split('\n')[0].strip()
            target_sentiment = sentiment_line
        elif 'very negative' in input_text.lower():
            target_sentiment = 'very negative'
        elif 'very positive' in input_text.lower():
            target_sentiment = 'very positive'
        elif 'negative' in input_text.lower():
            target_sentiment = 'negative'
        elif 'positive' in input_text.lower():
            target_sentiment = 'positive'
        
        grpo_item = {
            'sentiment': target_sentiment,
            'generated_sentiment': target_sentiment,
            'review_text': item.get('output', ''),
            'original_input': item.get('input', ''),
            'prompt': full_prompt
        }
        grpo_data.append(grpo_item)
    
    print(f"✅ GRPO dataset prepared with {len(grpo_data)} samples")
    return Dataset.from_list(grpo_data)

class SynthesisRewardCalculator:
    """Reward calculator for synthesis tasks."""
    
    def __init__(self, readability_model_path, device='cuda'):
        self.device = device
        self.readability_model_path = readability_model_path
        print(f"✅ SynthesisRewardCalculator initialized (device: {device})")

def setup_reward_calculators():
    """Set up all reward calculators."""
    print("🔧 Initializing reward enhancement system...")
    
    attr_loader = AttrPromptAttributeLoader(ATTRPROMPT_CONFIG['base_path'])
    compliance_calculator = AttributeComplianceCalculator(attr_loader)
    reward_calculator = SynthesisRewardCalculator(READABILITY_MODEL_PATH)
    
    print("✅ Reward enhancement calculators initialized")
    return reward_calculator, None, attr_loader, compliance_calculator

def plot_enhanced_training_curves(trainer, output_dir):
    """Plot enhanced training curves and save metrics."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    try:
        log_history = trainer.state.log_history
        
        if not log_history:
            print("⚠️ No training logs found for plotting")
            return
        
        # Extract metrics from training log
        steps = []
        total_rewards = []
        losses = []
        sentiment_rewards = []
        attribute_rewards = []
        # length_rewards = []  # Length reward removed
        quality_rewards = []
        diversity_rewards = []
        
        for log_entry in log_history:
            if 'step' in log_entry:
                steps.append(log_entry['step'])
                total_rewards.append(log_entry.get('reward', 0))
                losses.append(log_entry.get('loss', 0))
                
                sentiment_rewards.append(log_entry.get('rewards/reward_sentiment_consistency/mean', 0))
                attribute_rewards.append(log_entry.get('rewards/reward_attribute_compliance/mean', 0))
                # length_rewards.append(log_entry.get('rewards/reward_length_compliance/mean', 0))  # Length reward removed
                quality_rewards.append(log_entry.get('rewards/reward_generation_quality/mean', 0))
                diversity_rewards.append(log_entry.get('rewards/reward_batch_diversity/mean', 0))
        
        if not steps:
            print("⚠️ No step data found for plotting")
            return
        
        # Configure plot layout (2x3, four reward components plus loss and total)
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Reward Enhanced GRPO Training Progress (4 Rewards)', fontsize=16, fontweight='bold')
        
        # 1. Total reward curve
        axes[0, 0].plot(steps, total_rewards, 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_title('Total Reward (Enhanced)', fontweight='bold')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Loss curve
        axes[0, 1].plot(steps, losses, 'r-', linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_title('Training Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sentiment consistency reward
        axes[0, 2].plot(steps, sentiment_rewards, 'g-', linewidth=2, marker='^', markersize=3)
        axes[0, 2].set_title('Sentiment Consistency (Max: 1.0)', fontweight='bold')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Reward')
        axes[0, 2].set_ylim(0, 1.1)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Attribute compliance reward
        axes[1, 0].plot(steps, attribute_rewards, 'm-', linewidth=2, marker='d', markersize=3)
        axes[1, 0].set_title('Attribute Compliance (Max: 1.0)', fontweight='bold')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Generation quality reward (includes length)
        axes[1, 1].plot(steps, quality_rewards, 'orange', linewidth=2, marker='*', markersize=4)
        axes[1, 1].set_title('Generation Quality (Max: 1.0, includes length)', fontweight='bold')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Batch diversity reward
        axes[1, 2].plot(steps, diversity_rewards, 'cyan', linewidth=2, marker='v', markersize=3)
        axes[1, 2].set_title('Batch Diversity (Max: 1.0, lowered threshold)', fontweight='bold')
        axes[1, 2].set_xlabel('Training Step')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].set_ylim(0, 1.1)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot image
        output_path = f"{output_dir}/enhanced_training_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Enhanced training curves saved to: {output_path}")
        
        # Save metrics to CSV (length reward column removed)
        df = pd.DataFrame({
            'Step': steps,
            'Total_Reward': total_rewards,
            'Loss': losses,
            'Sentiment_Reward': sentiment_rewards,
            'Attribute_Reward': attribute_rewards,
            'Quality_Reward': quality_rewards,
            'Diversity_Reward': diversity_rewards
        })
        
        csv_path = f"{output_dir}/enhanced_training_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Enhanced training metrics saved to: {csv_path}")
        
    except Exception as e:
        print(f"❌ Error plotting enhanced training curves: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main training pipeline for reward-enhanced GRPO on Yelp."""
    print("🌟 Starting reward-enhanced GRPO training...")
    print("🎯 Core strategy: relax thresholds, raise reward ceiling, and encourage learning")
    print("📊 Config: {} samples, {} epochs".format(ENHANCED_CONFIG["max_train_samples"], ENHANCED_CONFIG["num_train_epochs"]))
    
    print("🎯 Reward enhancement design:")
    print("   Sentiment consistency: max 1.0 (map [0, 0.5] → [0, 1.0])")
    print("   Attribute compliance: max 1.0 (high score if keywords present)")
    print("   Generation quality: max 1.0 (includes length and JSON checks)")
    print("   Batch diversity: max 1.0 (lower thresholds to fit fixed prompts)")
    
    print("🔧 Training tweaks:")
    print(f"   Generations per prompt: {ENHANCED_CONFIG['num_generations']} (more candidates)")
    print(f"   Learning rate: {ENHANCED_CONFIG['learning_rate']} (slightly higher)")
    print(f"   Warmup steps: {ENHANCED_CONFIG['warmup_steps']} (more warmup)")
    print(f"   Temperature: {GENERATION_CONFIG['temperature']} (slightly more creative)")
    
    # Initialize training visualizer
    visualizer = initialize_visualizer(OUTPUT_DIR)
    
    # Set up reward calculators
    reward_calculator, novelsum_calculator, attr_loader, compliance_calculator = setup_reward_calculators()
    
    # Create dataset
    dataset, training_data_global = create_optimized_dataset(
        DATA_FILE, 
        ENHANCED_CONFIG['max_train_samples'], 
        ENHANCED_CONFIG['per_device_train_batch_size']
    )
    
    # Initialize global variables for reward modules
    initialize_reward_globals(
        training_data_global, 
        ENHANCED_CONFIG['per_device_train_batch_size'],
        reward_calculator, 
        novelsum_calculator,
        attr_loader, 
        compliance_calculator,
        optimized_sample_config=ENHANCED_SAMPLE_REWARDS_CONFIG,
        optimized_batch_config=ENHANCED_BATCH_REWARDS_CONFIG,
        embedding_model_path=EMBEDDING_MODEL_PATH
    )
    
    set_training_visualizer(visualizer)
    
    # Check and load pre-merged SFT model
    if USE_MERGED_MODEL and os.path.exists(MERGED_MODEL_PATH):
        print("🤖 Loading pre-merged full SFT model...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("✅ Pre-merged full SFT model loaded successfully")
        
    else:
        print("❌ Pre-merged model does not exist!")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Configure generation parameters on the model
    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
        model.generation_config.top_p = GENERATION_CONFIG["top_p"]
        model.generation_config.top_k = GENERATION_CONFIG["top_k"]
        model.generation_config.repetition_penalty = GENERATION_CONFIG["repetition_penalty"]
        model.generation_config.temperature = GENERATION_CONFIG["temperature"]
        model.generation_config.max_new_tokens = ENHANCED_CONFIG["max_completion_length"]
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        print("🔥 Reward-enhanced generation config set")
    
    # Create enhanced reward functions
    reward_functions = create_enhanced_reward_functions()
    print("✅ Reward-enhanced functions created")
    
    reward_weights = [
        ENHANCED_SAMPLE_REWARDS_CONFIG['sentiment_consistency_weight'],
        ENHANCED_SAMPLE_REWARDS_CONFIG['attribute_compliance_weight'],
        ENHANCED_SAMPLE_REWARDS_CONFIG['generation_quality_weight'],
        ENHANCED_BATCH_REWARDS_CONFIG['batch_diversity_weight'],
    ]
    
    print(f"📊 Reward weights: {reward_weights}")
    print(f"📊 Sum of weights: {sum(reward_weights)}")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM", inference_mode=False
    )
    
    # GRPO training configuration
    grpo_config = GRPOConfig(
        learning_rate=ENHANCED_CONFIG["learning_rate"],
        num_train_epochs=ENHANCED_CONFIG["num_train_epochs"],
        per_device_train_batch_size=ENHANCED_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=ENHANCED_CONFIG["gradient_accumulation_steps"],
        logging_steps=ENHANCED_CONFIG["logging_steps"],
        save_steps=ENHANCED_CONFIG["save_steps"],
        warmup_steps=ENHANCED_CONFIG["warmup_steps"],
        max_grad_norm=ENHANCED_CONFIG["max_grad_norm"],
        dataloader_num_workers=ENHANCED_CONFIG["dataloader_num_workers"],
        output_dir=OUTPUT_DIR,
        num_generations=ENHANCED_CONFIG["num_generations"],
        max_completion_length=ENHANCED_CONFIG["max_completion_length"],
        reward_weights=reward_weights,
        temperature=GENERATION_CONFIG["temperature"],
        top_p=GENERATION_CONFIG["top_p"],
        top_k=GENERATION_CONFIG["top_k"],
        repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
        remove_unused_columns=False,
        report_to=[],
    )
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    print("🔥 Starting reward-enhanced GRPO training...")
    print("📋 Enhancements:")
    print("   - 🎯 Relax scoring standards for all rewards")
    print("   - 📈 Raise reward ceiling for stronger signal")
    print("   - 🏆 Add base reward to reduce zero scores")
    print("   - 🚀 Optimize training hyper-parameters")
    print("   - 🎨 Increase diversity and creativity of generations")
    print("-" * 80)
    
    # Start training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    training_time = (end_time - start_time) / 60
    print(f"⏱️ Training time: {training_time:.2f} minutes")
    
    # Save final model
    print("💾 Saving trained model...")
    trainer.save_model()
    
    # Plot training curves
    print("📊 Plotting reward-enhanced training curves...")
    plot_enhanced_training_curves(trainer, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("🎉 Reward-enhanced GRPO training finished!")
    print(f"   Training time: {training_time:.2f} minutes")
    print(f"   Samples processed: {ENHANCED_CONFIG['max_train_samples']}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("   Reward enhancements:")
    print("     - ✅ Unified reward upper bound 1.0 for easier analysis")
    print("     - ✅ Attribute reward: high score when keywords present")
    print("     - ✅ Generation quality: combines length and JSON checks")
    print("     - ✅ Diversity reward: lower thresholds for fixed prompts")
    print("     - ✅ Removed standalone length reward to avoid duplication")
    print("=" * 80)

if __name__ == "__main__":
    main()
