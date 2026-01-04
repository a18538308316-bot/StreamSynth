#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen版本 奖励增强版GRPO训练脚本 - Amazon数据集
在保持原LLaMA脚本奖励/数据处理逻辑不变的情况下，仅替换底层基础模型为Qwen2.5-7B-instruct。

可通过 --base-model-path 覆盖默认的合并模型路径，方便之后替换为RL继续训练的中间检查点。
"""

import os
import sys
import time
import json
import re
import torch
import numpy as np
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset

from scripts_amazon.data_utils import create_optimized_dataset
from scripts_amazon.reward_functions import (
    initialize_reward_globals,
    set_training_visualizer,
)
from scripts_amazon.training_visualizer import initialize_visualizer
from scripts_amazon.batch_diversity_reward import initialize_batch_diversity_calculator
from scripts_amazon.amazon_attribute_config import (
    AMAZON_ATTRPROMPT_CONFIG,
    load_amazon_sample_attributes
)

# =============================================================================
# 路径与常量配置（保持与原脚本一致，仅模型路径改为Qwen）
# =============================================================================

DATA_FILE = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/amazon/amazon_train_4000.json"
# 默认使用先前合并的SFT模型目录；可通过命令行覆盖
MERGED_MODEL_PATH = ""
OUTPUT_DIR = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/TRL-GRPO-ohter-dataset/amazon/qwen_grpo_reward_enhanced_output"

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

ENHANCED_BATCH_REWARDS_CONFIG = {
    "batch_diversity_weight": 0.3,
}

GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.85,
    "top_p": 0.92,
    "top_k": 60,
    "repetition_penalty": 1.12,
    "pad_token_id": None,
}

# =============================================================================
# 增强奖励函数包装 (与LLaMA版本保持一致)
# =============================================================================

def create_enhanced_reward_functions():
    from scripts_amazon.reward_functions import reward_generation_quality_batch
    from scripts_amazon.batch_diversity_reward import reward_batch_diversity

    sentiment_levels = {
        'very negative': 1,
        'negative': 2,
        'neutral': 3,
        'positive': 4,
        'very positive': 5,
    }

    negative_markers = {
        'very negative': ['terrible', 'awful', 'horrible', 'hate', 'worst', 'broken', 'refund', 'trash'],
        'negative': ['bad', 'poor', 'disappointed', 'issue', 'problem', 'mediocre', 'return'],
        'neutral': ['okay', 'average', 'fine', 'decent', 'mixed', 'acceptable'],
        'positive': ['good', 'great', 'satisfied', 'happy', 'works well', 'recommend'],
        'very positive': ['excellent', 'amazing', 'fantastic', 'outstanding', 'love', 'perfect'],
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

    def _clean_review_text(input_value):
        if not isinstance(input_value, str):
            return ""
        cleaned = input_value.strip()
        if cleaned.lower().startswith('text:'):
            cleaned = cleaned[5:].strip()
        return cleaned

    def _infer_sentiment_from_text(text):
        text_lower = text.lower()
        best_match = ('neutral', 0)
        for label, keywords in negative_markers.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > best_match[1]:
                best_match = (label, matches)
        return best_match[0]

    def _extract_target_sentiment(prompt):
        if not isinstance(prompt, str):
            return 'neutral'
        sentiment_patterns = [
            r'Target sentiment label \(MUST match exactly\):\s*([^\n]+)',
            r'Target sentiment for generation:\s*([^\n]+)',
            r'Sentiment label \(must use exactly one\):\s*"?([^"\n]+)"?',
        ]
        prompt_lower = prompt.lower()
        for pattern in sentiment_patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().lower()
        for label in sentiment_levels:
            if label in prompt_lower:
                return label
        return 'neutral'

    def _score_sentiment_match(target_label, completion_json, completion_text):
        target_label = target_label.lower()
        candidate_label = None
        if completion_json and isinstance(completion_json.get('output'), str):
            candidate_label = completion_json['output'].strip().lower()
        if not candidate_label:
            candidate_label = _infer_sentiment_from_text(completion_text)
        target_level = sentiment_levels.get(target_label, 3)
        candidate_level = sentiment_levels.get(candidate_label, None)
        if candidate_level is None:
            candidate_level = sentiment_levels.get(_infer_sentiment_from_text(completion_text), 3)
        diff = abs(candidate_level - target_level)
        if diff == 0:
            return 1.0
        if diff == 1:
            return 0.6
        if diff == 2:
            return 0.25
        return 0.0

    def _extract_attributes(prompt):
        attributes = load_amazon_sample_attributes(prompt or "")
        normalized = {k: v for k, v in attributes.items() if isinstance(v, str) and v}
        return normalized

    reward_call_counter = 0
    current_training_step = 0

    def enhanced_sentiment_consistency(completions, **kwargs):
        nonlocal reward_call_counter, current_training_step
        reward_call_counter += 1
        current_training_step = reward_call_counter // 4
        kwargs['step'] = current_training_step
        prompts = kwargs.get('prompts', [])
        rewards = []
        for idx, completion in enumerate(completions):
            prompt = prompts[idx] if idx < len(prompts) else ""
            target_sentiment = _extract_target_sentiment(prompt)
            parsed_json = _extract_first_json_object(completion)
            review_text = _clean_review_text(parsed_json.get('input')) if parsed_json else completion
            reward = _score_sentiment_match(target_sentiment, parsed_json, review_text)
            rewards.append(float(np.clip(reward, 0.0, 1.0)))
        return rewards

    def enhanced_attribute_compliance(completions, **kwargs):
        kwargs['step'] = current_training_step
        prompts = kwargs.get('prompts', [])
        rewards = []
        for idx, completion in enumerate(completions):
            prompt = prompts[idx] if idx < len(prompts) else ""
            attributes = _extract_attributes(prompt)
            target_sentiment = _extract_target_sentiment(prompt)
            parsed_json = _extract_first_json_object(completion)
            review_text = _clean_review_text(parsed_json.get('input')) if parsed_json else completion
            if not attributes or not review_text:
                rewards.append(0.0)
                continue
            total_score = 0.0
            count = 0
            for attr_name, target_value in attributes.items():
                check_func = AMAZON_ATTRPROMPT_CONFIG.get_attribute_check_function(
                    attr_name, target_value, sentiment=target_sentiment
                )
                try:
                    attr_score = float(check_func(review_text))
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
                reward = 0.1
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
# 数据处理函数 (保持原逻辑)
# =============================================================================

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
        output_text = item.get('output', '')
        target_sentiment = 'neutral'
        try:
            output_json = json.loads(output_text)
            if 'output' in output_json:
                target_sentiment = output_json['output']
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        if target_sentiment == 'neutral':
            if 'Target sentiment label (MUST match exactly):' in input_text:
                sentiment_line = input_text.split('Target sentiment label (MUST match exactly):')[1].split('\n')[0].strip()
                target_sentiment = sentiment_line
            elif 'Target sentiment for generation:' in input_text:
                sentiment_line = input_text.split('Target sentiment for generation:')[1].split('\n')[0].strip()
                target_sentiment = sentiment_line
        if target_sentiment == 'neutral':
            if 'very negative' in input_text.lower():
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
    print(f"✅ GRPO数据集准备完成，共{len(grpo_data)}个样本")
    return Dataset.from_list(grpo_data)

class SynthesisRewardCalculator:
    def __init__(self, readability_model_path, device='cuda'):
        self.device = device
        self.readability_model_path = readability_model_path
        print(f"✅ SynthesisRewardCalculator初始化完成 (设备: {device})")

def setup_reward_calculators():
    print("🔧 初始化Amazon数据集奖励增强系统 (Qwen版本)...")
    amazon_config = AMAZON_ATTRPROMPT_CONFIG
    reward_calculator = SynthesisRewardCalculator(READABILITY_MODEL_PATH)
    print("✅ Amazon奖励增强计算器初始化完成")
    print(f"📋 支持的属性: {list(amazon_config.attributes.keys())}")
    print(f"📋 支持的情感标签: {amazon_config.sentiment_labels}")
    return reward_calculator, None, amazon_config, None

def plot_enhanced_training_curves(trainer, output_dir):
    import matplotlib.pyplot as plt
    import pandas as pd
    try:
        log_history = trainer.state.log_history
        if not log_history:
            print("⚠️ No training logs found for plotting")
            return
        steps = []
        total_rewards = []
        losses = []
        sentiment_rewards = []
        attribute_rewards = []
        quality_rewards = []
        diversity_rewards = []
        for log_entry in log_history:
            if 'step' in log_entry:
                steps.append(log_entry['step'])
                total_rewards.append(log_entry.get('reward', 0))
                losses.append(log_entry.get('loss', 0))
                sentiment_rewards.append(log_entry.get('rewards/reward_sentiment_consistency/mean', 0))
                attribute_rewards.append(log_entry.get('rewards/reward_attribute_compliance/mean', 0))
                quality_rewards.append(log_entry.get('rewards/reward_generation_quality/mean', 0))
                diversity_rewards.append(log_entry.get('rewards/reward_batch_diversity/mean', 0))
        if not steps:
            print("⚠️ No step data found for plotting")
            return
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Qwen Reward Enhanced GRPO Training Progress (Amazon)', fontsize=16, fontweight='bold')
        axes[0,0].plot(steps, total_rewards, 'b-', linewidth=2, marker='o', markersize=3)
        axes[0,0].set_title('Total Reward (Enhanced)', fontweight='bold'); axes[0,0].set_xlabel('Step'); axes[0,0].set_ylabel('Reward'); axes[0,0].grid(True, alpha=0.3)
        axes[0,1].plot(steps, losses, 'r-', linewidth=2, marker='s', markersize=3)
        axes[0,1].set_title('Training Loss', fontweight='bold'); axes[0,1].set_xlabel('Step'); axes[0,1].set_ylabel('Loss'); axes[0,1].grid(True, alpha=0.3)
        axes[0,2].plot(steps, sentiment_rewards, 'g-', linewidth=2, marker='^', markersize=3)
        axes[0,2].set_title('Sentiment Consistency (≤1.0)', fontweight='bold'); axes[0,2].set_xlabel('Step'); axes[0,2].set_ylabel('Reward'); axes[0,2].set_ylim(0,1.1); axes[0,2].grid(True, alpha=0.3)
        axes[1,0].plot(steps, attribute_rewards, 'm-', linewidth=2, marker='d', markersize=3)
        axes[1,0].set_title('Attribute Compliance (≤1.0)', fontweight='bold'); axes[1,0].set_xlabel('Step'); axes[1,0].set_ylabel('Reward'); axes[1,0].set_ylim(0,1.1); axes[1,0].grid(True, alpha=0.3)
        axes[1,1].plot(steps, quality_rewards, 'orange', linewidth=2, marker='*', markersize=4)
        axes[1,1].set_title('Generation Quality (≤1.0)', fontweight='bold'); axes[1,1].set_xlabel('Step'); axes[1,1].set_ylabel('Reward'); axes[1,1].set_ylim(0,1.1); axes[1,1].grid(True, alpha=0.3)
        axes[1,2].plot(steps, diversity_rewards, 'cyan', linewidth=2, marker='v', markersize=3)
        axes[1,2].set_title('Batch Diversity (≤1.0)', fontweight='bold'); axes[1,2].set_xlabel('Step'); axes[1,2].set_ylabel('Reward'); axes[1,2].set_ylim(0,1.1); axes[1,2].grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = f"{output_dir}/qwen_enhanced_training_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(); print(f"✅ Curves saved: {output_path}")
        import pandas as pd
        df = pd.DataFrame({
            'Step': steps,
            'Total_Reward': total_rewards,
            'Loss': losses,
            'Sentiment_Reward': sentiment_rewards,
            'Attribute_Reward': attribute_rewards,
            'Quality_Reward': quality_rewards,
            'Diversity_Reward': diversity_rewards,
        })
        csv_path = f"{output_dir}/qwen_enhanced_training_metrics.csv"
        df.to_csv(csv_path, index=False); print(f"✅ Metrics saved: {csv_path}")
    except Exception as e:
        print(f"❌ Plot error: {e}")
        import traceback; traceback.print_exc()

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Amazon Qwen GRPO reward enhanced training", add_help=True)
    # 关键修改1：设为可选参数，默认值为空字符串（不强制必填）
    parser.add_argument("--base-model-path", type=str, default="",
                      help="模型路径（从外部传入）")
    # 其他参数保持不变
    parser.add_argument("--enable-dynamic-reward", action="store_true", help="启用动态连续奖励缩放与trace日志记录")
    parser.add_argument("--reward-trace-file", type=str, default=None, help="自定义奖励trace JSONL文件路径")
    parser.add_argument("--dynamic-warmup-steps", type=int, default=300, help="动态奖励权重warmup步数")
    parser.add_argument("--dynamic-plateau-shift-steps", type=int, default=3000, help="动态奖励权重周期/平台漂移步数")
    parser.add_argument("--dynamic-ma-decay", type=float, default=0.98, help="移动平均跟踪衰减系数")
    parser.add_argument("--dry-run", action="store_true", help="仅初始化各组件，不执行trainer.train()")
    parser.add_argument("--override-max-train-samples", type=int, default=None, help="覆盖配置中的max_train_samples")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"⚠️ 忽略未知参数: {unknown}")
    return args

def main():
    args = parse_args()
    global MERGED_MODEL_PATH
    
    # 关键修改2：优先使用外部注入的路径，其次用命令行参数（和Yelp脚本逻辑一致）
    if not MERGED_MODEL_PATH and args.base-model-path:
        MERGED_MODEL_PATH = args.base_model_path
    
    # 路径有效性校验（保留，避免无效路径）
    if not MERGED_MODEL_PATH or not os.path.exists(MERGED_MODEL_PATH):
        print(f"❌ 模型路径无效或未传入: {MERGED_MODEL_PATH}")
        return
    
    # 后续逻辑（打印日志、加载模型等）保持不变...
    
    print("🌟 启动Amazon Qwen奖励增强版GRPO训练...")
    print(f"🤖 使用外部传入模型路径: {MERGED_MODEL_PATH}")
    print("🌟 启动Amazon Qwen奖励增强版GRPO训练...")
    print(f"🤖 使用基础/合并模型: {MERGED_MODEL_PATH}")
    print("📊 配置: {}样本, {}轮".format(ENHANCED_CONFIG["max_train_samples"], ENHANCED_CONFIG["num_train_epochs"]))
    print("🏢 数据集: Amazon产品评论")
    print("🏷️  情感标签: very negative, negative, neutral, positive, very positive")

    visualizer = initialize_visualizer(OUTPUT_DIR)
    reward_calculator, novelsum_calculator, attr_loader, compliance_calculator = setup_reward_calculators()

    effective_max_samples = args.override_max_train_samples or ENHANCED_CONFIG['max_train_samples']
    if args.override_max_train_samples:
        print(f"⚙️ 覆盖max_train_samples: {ENHANCED_CONFIG['max_train_samples']} -> {effective_max_samples}")
    dataset, training_data_global = create_optimized_dataset(
        DATA_FILE,
        effective_max_samples,
        ENHANCED_CONFIG['per_device_train_batch_size']
    )

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

    if USE_MERGED_MODEL and os.path.exists(MERGED_MODEL_PATH):
        print("🤖 加载Qwen模型 (4bit量化)...")
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
            trust_remote_code=True  # Qwen需要
        )
        print(f"✅ Qwen模型加载成功 hidden_size={getattr(model.config,'hidden_size','N/A')} path={MERGED_MODEL_PATH}")
    else:
        print("❌ 模型路径不存在:", MERGED_MODEL_PATH)
        return

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
        model.generation_config.top_p = GENERATION_CONFIG["top_p"]
        model.generation_config.top_k = GENERATION_CONFIG["top_k"]
        model.generation_config.repetition_penalty = GENERATION_CONFIG["repetition_penalty"]
        model.generation_config.temperature = GENERATION_CONFIG["temperature"]
        model.generation_config.max_new_tokens = ENHANCED_CONFIG["max_completion_length"]
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        print("🔥 生成配置完成 (Qwen)")

    reward_functions = create_enhanced_reward_functions()
    print("✅ 奖励增强函数创建完成")

    dynamic_reward_enabled = False
    dynamic_config_summary = {}
    if args.enable_dynamic_reward:
        dynamic_reward_enabled = True
        try:
            from scripts_amazon.dynamic_reward_scaler import set_dynamic_reward_config, DEFAULT_BASE_WEIGHTS, DEFAULT_MAX_SCALE
            trace_file = args.reward_trace_file or os.path.join(OUTPUT_DIR, "dynamic_reward_trace.jsonl")
            set_dynamic_reward_config(
                trace_file=trace_file,
                enable_trace=True,
                warmup_steps=args.dynamic_warmup_steps,
                plateau_shift_steps=args.dynamic_plateau_shift_steps,
                decay=args.dynamic_ma_decay
            )
            dynamic_config_summary = {
                "trace_file": trace_file,
                "warmup_steps": args.dynamic_warmup_steps,
                "plateau_shift_steps": args.dynamic_plateau_shift_steps,
                "ma_decay": args.dynamic_ma_decay,
                "base_weights": DEFAULT_BASE_WEIGHTS,
                "max_scale": DEFAULT_MAX_SCALE
            }
            print("🔄 已启用动态连续奖励系统")
        except Exception as e:
            print(f"❌ 动态奖励系统启用失败: {e}")
            import traceback; traceback.print_exc()
            dynamic_reward_enabled = False

    reward_weights = [
        ENHANCED_SAMPLE_REWARDS_CONFIG['sentiment_consistency_weight'],
        ENHANCED_SAMPLE_REWARDS_CONFIG['attribute_compliance_weight'],
        ENHANCED_SAMPLE_REWARDS_CONFIG['generation_quality_weight'],
        ENHANCED_BATCH_REWARDS_CONFIG['batch_diversity_weight'],
    ]
    print(f"📊 增强版权重: {reward_weights}")
    print(f"📊 权重总和: {sum(reward_weights)}")

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM", inference_mode=False
    )

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

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        summary_path = os.path.join(OUTPUT_DIR, "qwen_enhanced_run_config_summary.json")
        summary_payload = {
            "enhanced_config": ENHANCED_CONFIG,
            "sample_reward_weights": ENHANCED_SAMPLE_REWARDS_CONFIG,
            "batch_reward_weights": ENHANCED_BATCH_REWARDS_CONFIG,
            "generation_config": GENERATION_CONFIG,
            "dynamic_reward_enabled": dynamic_reward_enabled,
            "dynamic_reward_config": dynamic_config_summary,
            "effective_max_samples": effective_max_samples,
            "base_model_path": MERGED_MODEL_PATH,
            "timestamp": time.time()
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        print(f"📝 配置概要已保存: {summary_path}")
    except Exception as e:
        print(f"⚠️ 配置概要保存失败: {e}")

    if args.dry_run:
        print("🚫 Dry-run模式: 跳过trainer.train()，仅做初始化验证。")
        return

    print("🔥 开始Qwen奖励增强GRPO训练...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f"⏱️ 训练耗时: {training_time:.2f}分钟")
    print("💾 保存训练后的模型...")
    trainer.save_model()
    print("📊 绘制训练曲线...")
    plot_enhanced_training_curves(trainer, OUTPUT_DIR)
    print("\n" + "="*80)
    print("🎉 Amazon数据集 Qwen 奖励增强GRPO训练完成！")
    print(f"   训练时长: {training_time:.2f}分钟")
    print(f"   处理样本: {ENHANCED_CONFIG['max_train_samples']}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("   奖励特性:")
    print("     - ✅ 情感一致性 (5类情感)")
    print("     - ✅ 属性符合度 (13属性)")
    print("     - ✅ 生成质量 (结构 & 长度)")
    print("     - ✅ 批次多样性")
    print("     - ✅ 统一奖励上限1.0")
    print("="*80)

if __name__ == "__main__":
    main()
