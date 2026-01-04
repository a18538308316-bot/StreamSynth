#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen reward-enhanced GRPO training script for the Yelp dataset.
Based on the original LLaMA version but using Qwen2.5-7B-instruct while keeping the reward logic aligned.
Supports --base-model-path to override the default model path.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset

sys.path.append('/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/TRL-GRPO/scripts_9_18')
from scripts_9_18.data_utils import create_optimized_dataset
from scripts_9_18.reward_functions import (
    initialize_reward_globals,
    set_training_visualizer,
)
from scripts_9_18.training_visualizer import initialize_visualizer
from scripts_9_18.batch_diversity_reward import initialize_batch_diversity_calculator
from scripts_9_18.attribute_handler import AttrPromptAttributeLoader, AttributeComplianceCalculator, ATTRPROMPT_CONFIG

DATA_FILE = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/train_data_4000.json"
MERGED_MODEL_PATH = ""
OUTPUT_DIR = "./qwen_yelp_grpo_reward_enhanced_output"
USE_MERGED_MODEL = True
READABILITY_MODEL_PATH = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/evaluate_model_data_continual_learning/reasoning-model"
EMBEDDING_MODEL_PATH = "/public/home/huzhenlin2023/synthetic_data/all-MiniLM-L6-v2"

ENHANCED_CONFIG = {
    "max_train_samples": 500,
    "num_train_epochs": 2.0,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,
    "max_completion_length": 800,
    "logging_steps": 5,
    "save_steps": 40,
    "learning_rate": 3e-6,
    "warmup_steps": 20,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 4,
}
ENHANCED_SAMPLE_REWARDS_CONFIG = {
    "sentiment_consistency_weight": 0.1,
    "attribute_compliance_weight": 0.1,
    "generation_quality_weight": 0.5,
}
ENHANCED_BATCH_REWARDS_CONFIG = {"batch_diversity_weight": 0.3}
GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.85,
    "top_p": 0.92,
    "top_k": 60,
    "repetition_penalty": 1.12,
    "pad_token_id": None,
}

def create_enhanced_reward_functions():
    from scripts_9_18.reward_functions import (
        reward_sentiment_consistency_batch,
        reward_attribute_compliance_batch,
        reward_generation_quality_batch,
        update_current_prompt_attributes
    )
    from scripts_9_18.batch_diversity_reward import reward_batch_diversity
    reward_call_counter = 0
    current_training_step = 0
    def enhanced_sentiment_consistency(completions, **kwargs):
        nonlocal reward_call_counter, current_training_step
        reward_call_counter += 1
        current_training_step = reward_call_counter // 4
        update_current_prompt_attributes(current_training_step)
        kwargs['step'] = current_training_step
        base_rewards = reward_sentiment_consistency_batch(completions, **kwargs)
        vals = base_rewards.tolist() if hasattr(base_rewards, 'tolist') else base_rewards
        return [min(v*2.0, 1.0) for v in vals]
    def enhanced_attribute_compliance(completions, **kwargs):
        kwargs['step'] = current_training_step
        base_rewards = reward_attribute_compliance_batch(completions, **kwargs)
        vals = base_rewards.tolist() if hasattr(base_rewards, 'tolist') else base_rewards
        out = []
        for r in vals:
            out.append(min(0.5 + r*2.0, 1.0) if r > 0 else 0.0)
        return out
    def enhanced_generation_quality(completions, **kwargs):
        kwargs['step'] = current_training_step
        base_rewards = reward_generation_quality_batch(completions, **kwargs)
        vals = base_rewards.tolist() if hasattr(base_rewards, 'tolist') else base_rewards
        out = []
        for r in vals:
            if r >= 0.15:
                out.append(min(0.4 + r*2.0, 1.0))
            elif r > 0:
                out.append(min(0.2 + r*1.5, 1.0))
            else:
                out.append(0.0)
        return out
    def enhanced_batch_diversity(completions, **kwargs):
        kwargs['step'] = current_training_step
        base_rewards = reward_batch_diversity(completions, **kwargs)
        vals = base_rewards.tolist() if hasattr(base_rewards, 'tolist') else (base_rewards if isinstance(base_rewards, list) else [base_rewards])
        out = []
        for r in vals:
            if r >= 0.3:
                out.append(min(0.8 + r*0.2, 1.0))
            elif r >= 0.2:
                out.append(min(0.5 + r*1.0, 1.0))
            elif r >= 0.1:
                out.append(min(0.3 + r*1.0, 1.0))
            else:
                out.append(0.1)
        return out
    enhanced_sentiment_consistency.__name__ = "reward_sentiment_consistency"
    enhanced_attribute_compliance.__name__ = "reward_attribute_compliance"
    enhanced_generation_quality.__name__ = "reward_generation_quality"
    enhanced_batch_diversity.__name__ = "reward_batch_diversity"
    return [enhanced_sentiment_consistency, enhanced_attribute_compliance, enhanced_generation_quality, enhanced_batch_diversity]

def load_and_process_data(data_file, max_samples=None):
    print(f"Loading synthesis data from {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    print(f"Loaded {len(data)} samples")
    grpo_data = []
    for item in data:
        full_prompt = item.get('instruction', '') + '\n\n' + item.get('input', '')
        input_text = item.get('input', '')
        target_sentiment = 'neutral'
        if 'Target sentiment for generation:' in input_text:
            target_sentiment = input_text.split('Target sentiment for generation:')[1].split('\n')[0].strip()
        elif 'very negative' in input_text.lower():
            target_sentiment = 'very negative'
        elif 'very positive' in input_text.lower():
            target_sentiment = 'very positive'
        elif 'negative' in input_text.lower():
            target_sentiment = 'negative'
        elif 'positive' in input_text.lower():
            target_sentiment = 'positive'
        grpo_data.append({
            'sentiment': target_sentiment,
            'generated_sentiment': target_sentiment,
            'review_text': item.get('output', ''),
            'original_input': item.get('input', ''),
            'prompt': full_prompt
        })
    print(f"✅ GRPO dataset prepared with {len(grpo_data)} samples")
    return Dataset.from_list(grpo_data)

class SynthesisRewardCalculator:
    def __init__(self, readability_model_path, device='cuda'):
        self.device = device
        self.readability_model_path = readability_model_path
        print(f"✅ SynthesisRewardCalculator initialized (device: {device})")

def setup_reward_calculators():
    print("🔧 Initializing reward enhancement system (Qwen Yelp)...")
    attr_loader = AttrPromptAttributeLoader(ATTRPROMPT_CONFIG['base_path'])
    compliance_calculator = AttributeComplianceCalculator(attr_loader)
    reward_calculator = SynthesisRewardCalculator(READABILITY_MODEL_PATH)
    print("✅ Reward enhancement calculators initialized")
    return reward_calculator, None, attr_loader, compliance_calculator

def plot_enhanced_training_curves(trainer, output_dir):
    import matplotlib.pyplot as plt, pandas as pd
    try:
        log_history = trainer.state.log_history
        if not log_history:
            print("⚠️ No training logs found for plotting"); return
        steps=[]; total_rewards=[]; losses=[]; sentiment_rewards=[]; attribute_rewards=[]; quality_rewards=[]; diversity_rewards=[]
        for log_entry in log_history:
            if 'step' in log_entry:
                steps.append(log_entry['step']); total_rewards.append(log_entry.get('reward',0)); losses.append(log_entry.get('loss',0))
                sentiment_rewards.append(log_entry.get('rewards/reward_sentiment_consistency/mean',0))
                attribute_rewards.append(log_entry.get('rewards/reward_attribute_compliance/mean',0))
                quality_rewards.append(log_entry.get('rewards/reward_generation_quality/mean',0))
                diversity_rewards.append(log_entry.get('rewards/reward_batch_diversity/mean',0))
        if not steps:
            print("⚠️ No step data found for plotting"); return
        plt.style.use('default'); fig, axes = plt.subplots(2,3, figsize=(18,12))
        fig.suptitle('Qwen Reward Enhanced GRPO (Yelp)', fontsize=16, fontweight='bold')
        axes[0,0].plot(steps,total_rewards,'b-',linewidth=2,marker='o',markersize=3); axes[0,0].set_title('Total Reward'); axes[0,0].grid(True,alpha=0.3)
        axes[0,1].plot(steps,losses,'r-',linewidth=2,marker='s',markersize=3); axes[0,1].set_title('Loss'); axes[0,1].grid(True,alpha=0.3)
        axes[0,2].plot(steps,sentiment_rewards,'g-',linewidth=2,marker='^',markersize=3); axes[0,2].set_title('Sentiment'); axes[0,2].set_ylim(0,1.1); axes[0,2].grid(True,alpha=0.3)
        axes[1,0].plot(steps,attribute_rewards,'m-',linewidth=2,marker='d',markersize=3); axes[1,0].set_title('Attribute'); axes[1,0].set_ylim(0,1.1); axes[1,0].grid(True,alpha=0.3)
        axes[1,1].plot(steps,quality_rewards,'orange',linewidth=2,marker='*',markersize=4); axes[1,1].set_title('Quality'); axes[1,1].set_ylim(0,1.1); axes[1,1].grid(True,alpha=0.3)
        axes[1,2].plot(steps,diversity_rewards,'cyan',linewidth=2,marker='v',markersize=3); axes[1,2].set_title('Diversity'); axes[1,2].set_ylim(0,1.1); axes[1,2].grid(True,alpha=0.3)
        plt.tight_layout(); out_path=f"{output_dir}/qwen_yelp_training_curves.png"; plt.savefig(out_path,dpi=300,bbox_inches='tight'); plt.close(); print(f"✅ Curves saved: {out_path}")
        import pandas as pd
        df=pd.DataFrame({'Step':steps,'Total_Reward':total_rewards,'Loss':losses,'Sentiment_Reward':sentiment_rewards,'Attribute_Reward':attribute_rewards,'Quality_Reward':quality_rewards,'Diversity_Reward':diversity_rewards})
        csv_path=f"{output_dir}/qwen_yelp_training_metrics.csv"; df.to_csv(csv_path,index=False); print(f"✅ Metrics saved: {csv_path}")
    except Exception as e:
        print(f"❌ Plot error: {e}"); import traceback; traceback.print_exc()

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Qwen Reward Enhanced GRPO Training")
    # Key change: make this optional with empty default (no hard-coded path)
    parser.add_argument("--base-model-path", type=str, default="", 
                      help="Base model path (provided externally)")
    parser.add_argument("--dry-run", action="store_true", help="Initialize only, skip training")
    parser.add_argument("--override-max-train-samples", type=int, default=None)
    args, unknown = parser.parse_known_args(argv)
    return args

def main():
    args = parse_args()
    global MERGED_MODEL_PATH
    
    # Priority: command-line argument first, then injected variable as fallback
    if args.base_model_path:
        MERGED_MODEL_PATH = args.base_model_path
    # If command-line argument is empty, rely on injected MERGED_MODEL_PATH (set by pipeline)
    
    # Validate that the model path exists
    if not MERGED_MODEL_PATH or not os.path.exists(MERGED_MODEL_PATH):
        print(f"❌ Invalid model path: {MERGED_MODEL_PATH}")
        return
    
    print(f"🌟 Starting training with model path: {MERGED_MODEL_PATH}")
    print("🌟 Starting Yelp Qwen reward-enhanced GRPO training...")
    visualizer=initialize_visualizer(OUTPUT_DIR)
    reward_calculator, novelsum_calculator, attr_loader, compliance_calculator = setup_reward_calculators()
    effective_max = args.override_max_train_samples or ENHANCED_CONFIG['max_train_samples']
    if args.override_max_train_samples: print(f"⚙️ Override max_train_samples -> {effective_max}")
    dataset, training_data_global = create_optimized_dataset(DATA_FILE, effective_max, ENHANCED_CONFIG['per_device_train_batch_size'])
    initialize_reward_globals(training_data_global, ENHANCED_CONFIG['per_device_train_batch_size'], reward_calculator, novelsum_calculator, attr_loader, compliance_calculator, optimized_sample_config=ENHANCED_SAMPLE_REWARDS_CONFIG, optimized_batch_config=ENHANCED_BATCH_REWARDS_CONFIG, embedding_model_path=EMBEDDING_MODEL_PATH)
    set_training_visualizer(visualizer)
    if USE_MERGED_MODEL and os.path.exists(MERGED_MODEL_PATH):
        print("🤖 Loading Qwen model (4bit)...")
        bnb_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
        model=AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        print(f"✅ Qwen model loaded successfully hidden_size={getattr(model.config,'hidden_size', 'N/A')} path={MERGED_MODEL_PATH}")
    else:
        print("❌ Model path does not exist:", MERGED_MODEL_PATH); return
    tokenizer=AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token; tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.padding_side="left"
    if hasattr(model,'generation_config'):
        gc=model.generation_config; gc.do_sample=True; gc.top_p=GENERATION_CONFIG['top_p']; gc.top_k=GENERATION_CONFIG['top_k']; gc.repetition_penalty=GENERATION_CONFIG['repetition_penalty']; gc.temperature=GENERATION_CONFIG['temperature']; gc.max_new_tokens=ENHANCED_CONFIG['max_completion_length']; gc.pad_token_id=tokenizer.eos_token_id; gc.eos_token_id=tokenizer.eos_token_id
        print("🔥 Generation configuration set")
    reward_functions=create_enhanced_reward_functions(); print("✅ Reward functions ready")
    reward_weights=[ENHANCED_SAMPLE_REWARDS_CONFIG['sentiment_consistency_weight'], ENHANCED_SAMPLE_REWARDS_CONFIG['attribute_compliance_weight'], ENHANCED_SAMPLE_REWARDS_CONFIG['generation_quality_weight'], ENHANCED_BATCH_REWARDS_CONFIG['batch_diversity_weight']]
    print("📊 Reward weights:", reward_weights)
    lora_config=LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],lora_dropout=0.1,bias="none",task_type="CAUSAL_LM",inference_mode=False)
    grpo_config=GRPOConfig(learning_rate=ENHANCED_CONFIG['learning_rate'],num_train_epochs=ENHANCED_CONFIG['num_train_epochs'],per_device_train_batch_size=ENHANCED_CONFIG['per_device_train_batch_size'],gradient_accumulation_steps=ENHANCED_CONFIG['gradient_accumulation_steps'],logging_steps=ENHANCED_CONFIG['logging_steps'],save_steps=ENHANCED_CONFIG['save_steps'],warmup_steps=ENHANCED_CONFIG['warmup_steps'],max_grad_norm=ENHANCED_CONFIG['max_grad_norm'],dataloader_num_workers=ENHANCED_CONFIG['dataloader_num_workers'],output_dir=OUTPUT_DIR,num_generations=ENHANCED_CONFIG['num_generations'],max_completion_length=ENHANCED_CONFIG['max_completion_length'],reward_weights=reward_weights,temperature=GENERATION_CONFIG['temperature'],top_p=GENERATION_CONFIG['top_p'],top_k=GENERATION_CONFIG['top_k'],repetition_penalty=GENERATION_CONFIG['repetition_penalty'],remove_unused_columns=False,report_to=[])
    trainer=GRPOTrainer(model=model,reward_funcs=reward_functions,args=grpo_config,train_dataset=dataset,processing_class=tokenizer,peft_config=lora_config)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR,'qwen_yelp_run_config.json'),'w',encoding='utf-8') as f:
        json.dump({"enhanced_config":ENHANCED_CONFIG,"sample_reward_weights":ENHANCED_SAMPLE_REWARDS_CONFIG,"batch_reward_weights":ENHANCED_BATCH_REWARDS_CONFIG,"generation_config":GENERATION_CONFIG,"effective_max_samples":effective_max,"base_model_path":MERGED_MODEL_PATH,"timestamp":time.time()}, f, ensure_ascii=False, indent=2)
    if args.dry_run:
        print("🚫 Dry-run: skipping training")
        return
    print("🔥 Starting Qwen Yelp GRPO training...")
    st=time.time(); trainer.train(); et=time.time(); mins=(et-st)/60
    print(f"⏱️ Training time: {mins:.2f} minutes")
    print("💾 Saving model..."); trainer.save_model()
    print("📊 Plotting training curves..."); plot_enhanced_training_curves(trainer, OUTPUT_DIR)
    print("="*80); print("🎉 Yelp Qwen reward-enhanced training finished!"); print(f"Processed samples: {ENHANCED_CONFIG['max_train_samples']}"); print(f"Output directory: {OUTPUT_DIR}"); print("="*80)

if __name__ == '__main__':
    main()
