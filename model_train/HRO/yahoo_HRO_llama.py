#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward-enhanced GRPO training script for the Yahoo dataset (LLaMA-based).

This version relaxes reward thresholds and increases reward caps based on
production analysis, to better encourage positive rewards during training.
"""

import os
import sys
import time
import json
import re
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

# Import modular components
from scripts_yahoo.data_utils import create_optimized_dataset
from scripts_yahoo.reward_functions import (
    initialize_reward_globals,
    set_training_visualizer,
)
from scripts_yahoo.training_visualizer import initialize_visualizer
from scripts_yahoo.batch_diversity_reward import initialize_batch_diversity_calculator

# Yahoo dataset specific reward configuration and attribute helpers
from scripts_yahoo.yahoo_attribute_config import (
    YahooAttrPromptConfig, 
    YAHOO_ATTRPROMPT_CONFIG,
    load_yahoo_sample_attributes,
    get_yahoo_labels
)
# =============================================================================
# Reward configuration
# =============================================================================

DATA_FILE = "/path/to/yahoo_train_1500.json"
MERGED_MODEL_PATH = "/path/to/yahoo_sft_grpo_sft_model"  # Yahoo-specific base model path
OUTPUT_DIR = "/path/to/yahoo_sft_grpo_enhanced_output"

USE_MERGED_MODEL = True

READABILITY_MODEL_PATH = "/path/to/reasoning-model"
EMBEDDING_MODEL_PATH = "/path/to/all-MiniLM-L6-v2"

# Reward-enhanced training configuration
ENHANCED_CONFIG = {
    "max_train_samples": 500,
    "num_train_epochs": 2.0,        # Slightly increase training epochs
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,           # More candidate generations per prompt
    "max_completion_length": 800,
    "logging_steps": 5,             
    "save_steps": 40,
    "learning_rate": 3e-6,          # Slightly higher learning rate
    "warmup_steps": 20,             # More warmup steps
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 4,
}

# 🎯 Re-tuned reward weight configuration
# Adjusted based on production performance observations
ENHANCED_SAMPLE_REWARDS_CONFIG = {
    "label_consistency_weight": 0.1,    # Label consistency weight (highest priority)
    "attribute_compliance_weight": 0.1,     # Attribute compliance weight (important)
    "generation_quality_weight": 0.5,       # Generation quality weight (includes length and JSON format)
    # Length reward is merged into generation quality
}

ENHANCED_BATCH_REWARDS_CONFIG = {
    "batch_diversity_weight": 0.3,          # Batch diversity weight
}

# Generation parameter configuration
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.85,            # Slightly higher creativity
    "top_p": 0.92,                  # Slightly looser nucleus sampling
    "top_k": 60,                    # Larger candidate vocabulary
    "repetition_penalty": 1.12,     # Slightly lower repetition penalty
    "pad_token_id": None,
}

# =============================================================================
# Reward-enhanced wrapper functions
# =============================================================================

def create_enhanced_reward_functions():
    """Create enhanced reward functions using parsed prompt and completion."""
    from scripts_yahoo.reward_functions import reward_generation_quality_batch
    from scripts_yahoo.batch_diversity_reward import reward_batch_diversity

    yahoo_labels = get_yahoo_labels()
    canonical_labels = {
        re.sub(r"[^a-z0-9]+", " ", label.lower()).strip(): label for label in yahoo_labels
    }

    label_keywords = {
        'society & culture': ['culture', 'society', 'tradition', 'community', 'cultural', 'custom', 'festival'],
        'science & mathematics': ['science', 'scientific', 'math', 'physics', 'chemistry', 'biology', 'research'],
        'health': ['health', 'medical', 'doctor', 'medicine', 'treatment', 'symptom', 'wellness'],
        'education & reference': ['education', 'school', 'learning', 'study', 'curriculum', 'student', 'teacher'],
        'computers & internet': ['computer', 'software', 'internet', 'technology', 'digital', 'online', 'cyber'],
        'sports': ['sport', 'athlete', 'team', 'league', 'game', 'match', 'season'],
        'business & finance': ['business', 'finance', 'market', 'investment', 'economy', 'stock', 'budget'],
        'entertainment & music': ['entertainment', 'music', 'movie', 'film', 'show', 'artist', 'album'],
        'family & relationships': ['family', 'relationship', 'parent', 'child', 'marriage', 'dating', 'partner'],
        'politics & government': ['politics', 'government', 'policy', 'election', 'law', 'political', 'governance'],
    }

    def _extract_first_json_object(text):
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            brace_idx = text.find('{', idx)
            if brace_idx == -1:
                break
            try:
                obj, end = decoder.raw_decode(text, brace_idx)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = brace_idx + 1
        return None

    def _clean_qa_text(input_value):
        if not isinstance(input_value, str):
            return ""
        content = input_value.strip()
        if content.lower().startswith('text:'):
            content = content[5:].strip()
        return content

    def _normalize_label(label):
        if not isinstance(label, str):
            return ""
        return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()

    def _label_tokens(label):
        normalized = _normalize_label(label)
        return set(normalized.split()) if normalized else set()

    def _infer_label_from_text(text):
        text_lower = text.lower()
        best_label = None
        best_score = 0
        for label, keywords in label_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_label = label
                best_score = score
        if best_score == 0:
            return ""
        normalized_key = _normalize_label(best_label)
        if normalized_key in canonical_labels:
            return canonical_labels[normalized_key]
        return best_label

    def _extract_target_label(prompt):
        if not isinstance(prompt, str):
            return 'Society & Culture'
        patterns = [
            r'Target topic label\s*\(.*?\):\s*([\w &/-]+)',
            r'Target topic label\s*:\s*([\w &/-]+)',
            r'Target label\s*:\s*([\w &/-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                normalized = _normalize_label(candidate)
                if normalized in canonical_labels:
                    return canonical_labels[normalized]
                if normalized:
                    for norm_label, original in canonical_labels.items():
                        if _label_tokens(norm_label) <= _label_tokens(candidate):
                            return original
                return candidate
        prompt_lower = prompt.lower()
        for label in yahoo_labels:
            if label.lower() in prompt_lower:
                return label
        return 'Society & Culture'

    def _extract_attributes(prompt):
        attributes = load_yahoo_sample_attributes(prompt or "")
        return {k: v for k, v in attributes.items() if isinstance(v, str) and v}

    def _score_label_match(target_label, candidate_label, qa_text):
        target_tokens = _label_tokens(target_label) or {'society', 'culture'}
        candidate_tokens = _label_tokens(candidate_label)

        def _similarity(tokens):
            if not tokens:
                return 0.0
            return len(target_tokens & tokens) / max(len(target_tokens), 1)

        candidate_sim = _similarity(candidate_tokens)
        if candidate_sim >= 0.99:
            return 1.0
        if candidate_sim >= 0.66:
            return 0.7
        if candidate_sim >= 0.33:
            return 0.4

        inferred_label = _infer_label_from_text(qa_text)
        inferred_tokens = _label_tokens(inferred_label)
        inferred_sim = _similarity(inferred_tokens)
        if inferred_sim >= 0.9:
            return 0.55
        if inferred_sim >= 0.5:
            return 0.3
        if inferred_sim >= 0.3:
            return 0.15
        return 0.0

    reward_call_counter = 0
    current_training_step = 0

    def enhanced_label_consistency(completions, **kwargs):
        nonlocal reward_call_counter, current_training_step
        reward_call_counter += 1
        current_training_step = reward_call_counter // 4
        kwargs['step'] = current_training_step

        prompts = kwargs.get('prompts', [])
        rewards = []
        for idx, completion in enumerate(completions):
            prompt = prompts[idx] if idx < len(prompts) else ""
            target_label = _extract_target_label(prompt)
            parsed_json = _extract_first_json_object(completion)
            qa_text = _clean_qa_text(parsed_json.get('input')) if parsed_json else completion
            candidate_label = parsed_json.get('output') if parsed_json else None
            reward = _score_label_match(target_label, candidate_label, qa_text)
            rewards.append(float(np.clip(reward, 0.0, 1.0)))
        return rewards

    def enhanced_attribute_compliance(completions, **kwargs):
        kwargs['step'] = current_training_step
        prompts = kwargs.get('prompts', [])
        rewards = []
        for idx, completion in enumerate(completions):
            prompt = prompts[idx] if idx < len(prompts) else ""
            attributes = _extract_attributes(prompt)
            target_label = _extract_target_label(prompt)
            parsed_json = _extract_first_json_object(completion)
            qa_text = _clean_qa_text(parsed_json.get('input')) if parsed_json else completion
            if not attributes or not qa_text:
                rewards.append(0.0)
                continue
            total_score = 0.0
            count = 0
            for attr_name, target_value in attributes.items():
                check_func = YAHOO_ATTRPROMPT_CONFIG.get_attribute_check_function(
                    attr_name, target_value, label=target_label
                )
                try:
                    attr_score = float(check_func(qa_text))
                except Exception:
                    attr_score = 0.0
                total_score += max(0.0, min(attr_score, 1.0))
                count += 1
            avg_score = total_score / count if count else 0.0
            if avg_score >= 0.75:
                reward = 1.0
            elif avg_score >= 0.55:
                reward = 0.7
            elif avg_score >= 0.35:
                reward = 0.4
            elif avg_score >= 0.15:
                reward = 0.15
            else:
                reward = 0.0
            rewards.append(float(reward))
        return rewards

    def enhanced_generation_quality(completions, **kwargs):
        kwargs['step'] = current_training_step
        base_rewards = reward_generation_quality_batch(completions, **kwargs)
        if hasattr(base_rewards, 'tolist'):
            return base_rewards.tolist()
        return [float(r) for r in base_rewards]

    def enhanced_batch_diversity(completions, **kwargs):
        kwargs['step'] = current_training_step
        rewards = reward_batch_diversity(completions, **kwargs)
        if hasattr(rewards, 'tolist'):
            return rewards.tolist()
        return [float(r) for r in rewards]

    enhanced_label_consistency.__name__ = "reward_label_consistency"
    enhanced_attribute_compliance.__name__ = "reward_attribute_compliance"
    enhanced_generation_quality.__name__ = "reward_generation_quality"
    enhanced_batch_diversity.__name__ = "reward_batch_diversity"

    return [
        enhanced_label_consistency,
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
        output_text = item.get('output', '')
        target_label = 'Society & Culture'  # Default label
        
        # Method 1: extract label from JSON in the output field (most accurate)
        try:
            import json
            # Parse JSON from the output field
            output_json = json.loads(output_text)
            if 'output' in output_json:
                target_label = output_json['output']
        except (json.JSONDecodeError, KeyError, TypeError):
            # If JSON parsing fails, fall back to other methods
            pass

        # Method 2: if method 1 fails, extract from the input field
        if target_label == 'Society & Culture':
            if 'Target topic label:' in input_text:
                # Extract the content after "Target topic label:"
                label_line = input_text.split('Target topic label:')[1].split('\n')[0].strip()
                target_label = label_line
            elif 'Target label:' in input_text:
                # Compatible with alternative formats
                label_line = input_text.split('Target label:')[1].split('\n')[0].strip()
                target_label = label_line

        # Method 3: fallback, search Yahoo labels in the input text
        if target_label == 'Society & Culture':
            yahoo_labels = get_yahoo_labels()
            for label in yahoo_labels:
                if label.lower() in input_text.lower():
                    target_label = label
                    break
        
        grpo_item = {
            'label': target_label,
            'generated_label': target_label,
            'qa_text': item.get('output', ''),
            'original_input': item.get('input', ''),
            'prompt': full_prompt
        }
        grpo_data.append(grpo_item)
    
    print(f"✅ GRPO dataset prepared with {len(grpo_data)} samples")
    return Dataset.from_list(grpo_data)

class SynthesisRewardCalculator:
    """Reward calculator for the Yahoo data synthesis task."""
    
    def __init__(self, readability_model_path, device='cuda'):
        self.device = device
        self.readability_model_path = readability_model_path
        print(f"✅ SynthesisRewardCalculator initialized (device: {device})")

def setup_reward_calculators():
    """Set up Yahoo-specific reward calculators."""
    print("🔧 Initializing Yahoo reward-enhanced system...")
    
    # Use Yahoo-specific attribute configuration
    yahoo_config = YAHOO_ATTRPROMPT_CONFIG
    reward_calculator = SynthesisRewardCalculator(READABILITY_MODEL_PATH)
    
    print("✅ Yahoo reward calculator initialized")
    print(f"📋 Supported attributes: {list(yahoo_config.attributes.keys())}")
    print(f"📋 Supported labels: {yahoo_config.labels}")
    
    return reward_calculator, None, yahoo_config, None

def plot_enhanced_training_curves(trainer, output_dir):
    """Plot training curves for the reward-enhanced GRPO run."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    try:
        log_history = trainer.state.log_history
        
        if not log_history:
            print("⚠️ No training logs found for plotting")
            return
        
        # Extract data
        steps = []
        total_rewards = []
        losses = []
        label_rewards = []
        attribute_rewards = []
        quality_rewards = []
        diversity_rewards = []
        
        for log_entry in log_history:
            if 'step' in log_entry:
                steps.append(log_entry['step'])
                total_rewards.append(log_entry.get('reward', 0))
                losses.append(log_entry.get('loss', 0))
                
                label_rewards.append(log_entry.get('rewards/reward_label_consistency/mean', 0))
                attribute_rewards.append(log_entry.get('rewards/reward_attribute_compliance/mean', 0))
                quality_rewards.append(log_entry.get('rewards/reward_generation_quality/mean', 0))
                diversity_rewards.append(log_entry.get('rewards/reward_batch_diversity/mean', 0))
        
        if not steps:
            print("⚠️ No step data found for plotting")
            return
        
        # Configure plot style and layout
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Reward Enhanced GRPO Training Progress (4 Rewards) - Yahoo', fontsize=16, fontweight='bold')
        
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
        
        # 3. Label consistency reward
        axes[0, 2].plot(steps, label_rewards, 'g-', linewidth=2, marker='^', markersize=3)
        axes[0, 2].set_title('Label Consistency (Max: 1.0)', fontweight='bold')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Reward')
        axes[0, 2].set_ylim(0, 1.1)  # Set y-axis range
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Attribute compliance reward
        axes[1, 0].plot(steps, attribute_rewards, 'm-', linewidth=2, marker='d', markersize=3)
        axes[1, 0].set_title('Attribute Compliance (Max: 1.0)', fontweight='bold')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_ylim(0, 1.1)  # Set y-axis range
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Generation quality reward (includes length)
        axes[1, 1].plot(steps, quality_rewards, 'orange', linewidth=2, marker='*', markersize=4)
        axes[1, 1].set_title('Generation Quality (Max: 1.0, includes length)', fontweight='bold')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_ylim(0, 1.1)  # Set y-axis range
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Batch diversity reward
        axes[1, 2].plot(steps, diversity_rewards, 'cyan', linewidth=2, marker='v', markersize=3)
        axes[1, 2].set_title('Batch Diversity (Max: 1.0, lowered threshold)', fontweight='bold')
        axes[1, 2].set_xlabel('Training Step')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].set_ylim(0, 1.1)  # Set y-axis range
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = f"{output_dir}/enhanced_training_curves_yahoo.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Enhanced training curves saved to: {output_path}")
        
        # Save metrics to CSV
        df = pd.DataFrame({
            'Step': steps,
            'Total_Reward': total_rewards,
            'Loss': losses,
            'Label_Reward': label_rewards,
            'Attribute_Reward': attribute_rewards,
            'Quality_Reward': quality_rewards,
            'Diversity_Reward': diversity_rewards
        })
        
        csv_path = f"{output_dir}/enhanced_training_metrics_yahoo.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Enhanced training metrics saved to: {csv_path}")
        
    except Exception as e:
        print(f"❌ Error plotting enhanced training curves: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main training entry point for Yahoo reward-enhanced GRPO."""
    print("🌟 Starting Yahoo reward-enhanced GRPO training...")
    print("🎯 Core strategy: adapt to Yahoo QA generation with relaxed scoring and higher reward caps")
    print("📊 Config: {} samples, {} epochs".format(ENHANCED_CONFIG["max_train_samples"], ENHANCED_CONFIG["num_train_epochs"]))
    print("🏢 Dataset: Yahoo QA")
    print("🏷️  Labels: 10 Yahoo topic categories")
    print("⚙️  Attribute constraints: 8 Yahoo-specific attributes (question type, user intent, answer tone, etc.)")
    
    print("🎯 Reward enhancement strategy:")
    print(f"   Label consistency: max 1.0 (high score for exact match)")
    print(f"   Attribute compliance: max 1.0 (reward when related keywords appear)")
    print(f"   Generation quality: max 1.0 (includes length and QA-format checks)")
    print(f"   Batch diversity: max 1.0 (lower threshold to adapt to fixed prompts)")
    
    print("🔧 Training optimizations:")
    print(f"   Number of candidates: {ENHANCED_CONFIG['num_generations']} (more choices)")
    print(f"   Learning rate: {ENHANCED_CONFIG['learning_rate']} (slightly higher)")
    print(f"   Warmup steps: {ENHANCED_CONFIG['warmup_steps']} (more warmup)")
    print(f"   Temperature: {GENERATION_CONFIG['temperature']} (slightly higher creativity)")
    
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
    
    # Initialize global variables for Yahoo reward functions
    initialize_reward_globals(
        training_data_global, 
        ENHANCED_CONFIG['per_device_train_batch_size'],
        reward_calculator, 
        novelsum_calculator,
        attr_loader,  # yahoo_config object
        compliance_calculator,  # None for Yahoo
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
        print("❌ Pre-merged model path does not exist!")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Configure generation settings
    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
        model.generation_config.top_p = GENERATION_CONFIG["top_p"]
        model.generation_config.top_k = GENERATION_CONFIG["top_k"]
        model.generation_config.repetition_penalty = GENERATION_CONFIG["repetition_penalty"]
        model.generation_config.temperature = GENERATION_CONFIG["temperature"]
        model.generation_config.max_new_tokens = ENHANCED_CONFIG["max_completion_length"]
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        print("🔥 Reward-enhanced generation configuration set")
    
    # Create enhanced reward functions
    reward_functions = create_enhanced_reward_functions()
    print("✅ Reward-enhanced functions created")
    
    reward_weights = [
        ENHANCED_SAMPLE_REWARDS_CONFIG['label_consistency_weight'],
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
    
    # Configure GRPO training arguments
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
    print("📋 Enhancement summary:")
    print("   - 🎯 Relax all reward thresholds")
    print("   - 📈 Increase reward caps for stronger signals")
    print("   - 🏆 Boost base rewards to reduce zero scores")
    print("   - 🚀 Optimize training hyperparameters for efficiency")
    print("   - 🎨 Encourage generation diversity and creativity")
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
    print("🎉 Yahoo reward-enhanced GRPO training finished!")
    print(f"   Training time: {training_time:.2f} minutes")
    print(f"   Samples processed: {ENHANCED_CONFIG['max_train_samples']}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("   Yahoo-specific reward design:")
    print("     - ✅ Label consistency: 10 Yahoo topic labels")
    print("     - ✅ Attribute compliance: 8 Yahoo attribute constraints")
    print("     - ✅ Generation quality: Yahoo QA style and language quality")
    print("     - ✅ Diversity reward: batch-level content diversity")
    print("     - ✅ All rewards capped at 1.0 with relaxed thresholds to encourage learning")
    print("=" * 80)

if __name__ == "__main__":
    main()