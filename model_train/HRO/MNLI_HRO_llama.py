#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励增强版GRPO训练脚本 - 针对正向奖励的优化 (MNLI数据集)
基于production版本分析，放宽评分标准，提高奖励上限
"""

import os
import sys
import time
import json
import re
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset

from scripts_MNLI.data_utils import create_optimized_dataset
from scripts_MNLI.reward_functions import initialize_reward_globals, set_training_visualizer
from scripts_MNLI.training_visualizer import initialize_visualizer
from scripts_MNLI.mnli_attribute_config import MNLI_ATTRPROMPT_CONFIG, load_mnli_sample_attributes

# =============================================================================
# 奖励增强配置
# =============================================================================

DATA_FILE = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/MNLI/MNLI_train_1496.json"
MERGED_MODEL_PATH = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/TRL-GRPO-ohter-dataset/MNLI/merged_grpo_sft_mnli_model"
OUTPUT_DIR = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/TRL-GRPO-ohter-dataset/MNLI/MNLI_sft_grpo_enhanced_output-sim500-epo2-bat4-gr4-gen4"

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
    "label_consistency_weight": 0.1,
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
def create_enhanced_reward_functions():
    """创建增强版奖励函数，解析prompt/completion以获得真实标签与属性"""
    from scripts_MNLI.reward_functions import reward_generation_quality_batch
    from scripts_MNLI.batch_diversity_reward import reward_batch_diversity

    mnli_labels = {'entailment', 'contradiction', 'neutral'}

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

    def _clean_nli_text(input_value):
        if not isinstance(input_value, str):
            return ""
        return input_value.strip()

    def _split_premise_hypothesis(text):
        if not text:
            return "", ""
        match = re.search(r'Premise:\s*(.*?)\s*Hypothesis:\s*(.*)', text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            premise = match.group(1).strip()
            hypothesis = match.group(2).strip()
            return premise, hypothesis
        return text.strip(), ""

    def _extract_target_label(prompt):
        if not isinstance(prompt, str):
            return 'neutral'
        patterns = [
            r'Target label \(must match exactly\):\s*([^\n]+)',
            r'Target label:\s*([^\n]+)',
            r'label \(must match exactly\):\s*([^\n]+)',
            r'label:\s*([^\n]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().lower()
                if candidate in mnli_labels:
                    return candidate
        return 'neutral'

    def _extract_attributes(prompt):
        attributes = load_mnli_sample_attributes(prompt or "")
        return {k: v for k, v in attributes.items() if isinstance(v, (str, dict)) and v}

    def _has_negation(text):
        if not text:
            return False
        return bool(re.search(r"\b(no|not|never|none|cannot|can't|n't)\b", text.lower()))

    def _token_overlap(a, b):
        a_tokens = set(re.findall(r'\w+', a.lower()))
        b_tokens = set(re.findall(r'\w+', b.lower()))
        if not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(b_tokens)

    def _infer_label_from_text(premise, hypothesis):
        if not premise or not hypothesis:
            return 'neutral'
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        if hypothesis_lower in premise_lower or _token_overlap(premise, hypothesis) > 0.7 and _has_negation(premise) == _has_negation(hypothesis):
            return 'entailment'
        premise_neg = _has_negation(premise)
        hypothesis_neg = _has_negation(hypothesis)
        if _token_overlap(premise, hypothesis) > 0.4 and premise_neg != hypothesis_neg:
            return 'contradiction'
        if any(word in hypothesis_lower for word in ['cannot', "can't", 'never', 'no ']) and not premise_neg:
            return 'contradiction'
        return 'neutral'

    def _score_label_alignment(target_label, json_label, premise, hypothesis):
        json_label = json_label.lower() if isinstance(json_label, str) else None
        if json_label not in mnli_labels:
            json_label = None
        inferred_label = _infer_label_from_text(premise, hypothesis)
        if inferred_label == target_label:
            return 1.0 if json_label == target_label else 0.8
        if json_label == target_label:
            return 0.45
        if inferred_label in mnli_labels and json_label in mnli_labels:
            if {inferred_label, target_label} <= {'neutral', 'entailment'} or {inferred_label, target_label} <= {'neutral', 'contradiction'}:
                return 0.25
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
            json_label = parsed_json.get('output') if parsed_json else None
            qa_text = _clean_nli_text(parsed_json.get('input')) if parsed_json else completion
            premise, hypothesis = _split_premise_hypothesis(qa_text)
            reward = _score_label_alignment(target_label, json_label, premise, hypothesis)
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
            qa_text = _clean_nli_text(parsed_json.get('input')) if parsed_json else completion
            premise, hypothesis = _split_premise_hypothesis(qa_text)
            if not attributes or not qa_text:
                rewards.append(0.0)
                continue
            total_score = 0.0
            count = 0
            for attr_name, target_value in attributes.items():
                if attr_name == 'target_label':
                    continue
                check_func = MNLI_ATTRPROMPT_CONFIG.get_attribute_check_function(
                    attr_name, target_value, label=target_label
                )
                try:
                    if attr_name == 'length_premise':
                        attr_input = premise
                    elif attr_name == 'length_hypothesis':
                        attr_input = hypothesis
                    else:
                        attr_input = qa_text
                    attr_score = float(check_func(attr_input))
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
# 数据处理函数
# =============================================================================

def load_and_process_data(data_file, max_samples=None):
    """加载和处理训练数据"""
    print(f"Loading synthesis data from {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} samples")
    
    # 准备GRPO数据集
    grpo_data = []
    for item in data:
        full_prompt = item.get('instruction', '') + '\n\n' + item.get('input', '')
        
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        target_label = 'neutral'  # 默认标签
        
        # 方法1：从output字段的JSON中提取标签（最准确）
        try:
            import json
            # 解析output中的JSON
            output_json = json.loads(output_text)
            if 'output' in output_json:
                target_label = output_json['output']
        except (json.JSONDecodeError, KeyError, TypeError):
            # JSON解析失败，继续使用其他方法
            pass
        
        # 方法2：如果方法1失败，从input字段中提取
        if target_label == 'neutral':
            if 'Target label (must match exactly):' in input_text:
                # 提取Target label后面的内容
                label_line = input_text.split('Target label (must match exactly):')[1].split('\n')[0].strip()
                target_label = label_line
        
        # 验证标签是否有效
        if target_label not in ['entailment', 'contradiction', 'neutral']:
            target_label = 'neutral'  # 无效标签默认为neutral
        
        grpo_item = {
            'label': target_label,
            'generated_label': target_label,
            'nli_text': item.get('output', ''),
            'original_input': item.get('input', ''),
            'prompt': full_prompt
        }
        grpo_data.append(grpo_item)
    
    print(f"✅ GRPO数据集准备完成，共{len(grpo_data)}个样本")
    return Dataset.from_list(grpo_data)

class SynthesisRewardCalculator:
    """数据合成任务的奖励计算器"""
    
    def __init__(self, readability_model_path, device='cuda'):
        self.device = device
        self.readability_model_path = readability_model_path
        print(f"✅ SynthesisRewardCalculator初始化完成 (设备: {device})")

def setup_reward_calculators():
    """设置MNLI数据集专用奖励计算器"""
    print("🔧 初始化MNLI数据集奖励增强系统...")
    
    # 使用MNLI专用属性配置
    mnli_config = MNLI_ATTRPROMPT_CONFIG
    reward_calculator = SynthesisRewardCalculator(READABILITY_MODEL_PATH)
    
    print("✅ MNLI奖励增强计算器初始化完成")
    print(f"📋 支持的属性: {list(mnli_config.attributes.keys())}")
    print(f"📋 支持的标签: {mnli_config.labels}")
    
    return reward_calculator, None, mnli_config, None

def plot_enhanced_training_curves(trainer, output_dir):
    """绘制增强版训练曲线"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    try:
        log_history = trainer.state.log_history
        
        if not log_history:
            print("⚠️ No training logs found for plotting")
            return
        
        # 提取数据
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
        
        # 设置图表样式 (改为2x2布局，因为只有4个奖励函数)
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Reward Enhanced GRPO Training Progress (4 Rewards) - MNLI', fontsize=16, fontweight='bold')
        
        # 1. 总奖励曲线
        axes[0, 0].plot(steps, total_rewards, 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_title('Total Reward (Enhanced)', fontweight='bold')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 损失曲线
        axes[0, 1].plot(steps, losses, 'r-', linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_title('Training Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 标签一致性奖励
        axes[0, 2].plot(steps, label_rewards, 'g-', linewidth=2, marker='^', markersize=3)
        axes[0, 2].set_title('Label Consistency (Max: 1.0)', fontweight='bold')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Reward')
        axes[0, 2].set_ylim(0, 1.1)  # 设置y轴范围
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 属性符合度奖励
        axes[1, 0].plot(steps, attribute_rewards, 'm-', linewidth=2, marker='d', markersize=3)
        axes[1, 0].set_title('Attribute Compliance (Max: 1.0)', fontweight='bold')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_ylim(0, 1.1)  # 设置y轴范围
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 生成质量奖励 (包含长度)
        axes[1, 1].plot(steps, quality_rewards, 'orange', linewidth=2, marker='*', markersize=4)
        axes[1, 1].set_title('Generation Quality (Max: 1.0, includes length)', fontweight='bold')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_ylim(0, 1.1)  # 设置y轴范围
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 批次多样性奖励
        axes[1, 2].plot(steps, diversity_rewards, 'cyan', linewidth=2, marker='v', markersize=3)
        axes[1, 2].set_title('Batch Diversity (Max: 1.0, lowered threshold)', fontweight='bold')
        axes[1, 2].set_xlabel('Training Step')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].set_ylim(0, 1.1)  # 设置y轴范围
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = f"{output_dir}/enhanced_training_curves_mnli.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Enhanced training curves saved to: {output_path}")
        
        # 保存数据到CSV
        df = pd.DataFrame({
            'Step': steps,
            'Total_Reward': total_rewards,
            'Loss': losses,
            'Label_Reward': label_rewards,
            'Attribute_Reward': attribute_rewards,
            'Quality_Reward': quality_rewards,
            'Diversity_Reward': diversity_rewards
        })
        
        csv_path = f"{output_dir}/enhanced_training_metrics_mnli.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Enhanced training metrics saved to: {csv_path}")
        
    except Exception as e:
        print(f"❌ Error plotting enhanced training curves: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主训练流程"""
    print("🌟 启动MNLI数据集奖励增强版GRPO训练...")
    print("🎯 核心策略: 适配MNLI NLI生成，放宽评分标准，提高奖励上限，促进学习")
    print("📊 配置: {}样本, {}轮".format(ENHANCED_CONFIG["max_train_samples"], ENHANCED_CONFIG["num_train_epochs"]))
    print("🏢 数据集: MNLI自然语言推理")
    print("🏷️  标签: 3个MNLI分类标签")
    print("⚙️  属性约束: 8个MNLI特有属性（前提域、推理类型、语义现象等）")
    
    print("🎯 奖励增强策略:")
    print(f"   标签一致性: 上限1.0 (完全匹配给高分)")
    print(f"   属性符合度: 上限1.0 (有关键词就给高分)")
    print(f"   生成质量: 上限1.0 (包含长度和NLI格式检查)")
    print(f"   批次多样性: 上限1.0 (大幅降低阈值，适应固定提示词)")
    
    print("🔧 训练优化:")
    print(f"   生成候选数: {ENHANCED_CONFIG['num_generations']} (增加选择)")
    print(f"   学习率: {ENHANCED_CONFIG['learning_rate']} (稍微提高)")
    print(f"   预热步数: {ENHANCED_CONFIG['warmup_steps']} (增加预热)")
    print(f"   温度: {GENERATION_CONFIG['temperature']} (稍微提高创造性)")
    
    # 初始化训练可视化器
    visualizer = initialize_visualizer(OUTPUT_DIR)
    
    # 设置奖励计算器
    reward_calculator, novelsum_calculator, attr_loader, compliance_calculator = setup_reward_calculators()
    
    # 创建数据集
    dataset, training_data_global = create_optimized_dataset(
        DATA_FILE, 
        ENHANCED_CONFIG['max_train_samples'], 
        ENHANCED_CONFIG['per_device_train_batch_size']
    )
    
    # 初始化MNLI数据集奖励函数模块的全局变量
    initialize_reward_globals(
        training_data_global, 
        ENHANCED_CONFIG['per_device_train_batch_size'],
        reward_calculator, 
        novelsum_calculator,
        attr_loader,  # 这里是mnli_config
        compliance_calculator,  # 这里是None
        optimized_sample_config=ENHANCED_SAMPLE_REWARDS_CONFIG,
        optimized_batch_config=ENHANCED_BATCH_REWARDS_CONFIG,
        embedding_model_path=EMBEDDING_MODEL_PATH
    )
    
    set_training_visualizer(visualizer)
    
    # 检查并使用预先合并的模型
    if USE_MERGED_MODEL and os.path.exists(MERGED_MODEL_PATH):
        print("🤖 加载预先合并的完整SFT模型...")
        
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
        print("✅ 预先合并的完整SFT模型加载成功")
        
    else:
        print("❌ 预先合并的模型不存在!")
        return
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # 设置生成配置
    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
        model.generation_config.top_p = GENERATION_CONFIG["top_p"]
        model.generation_config.top_k = GENERATION_CONFIG["top_k"]
        model.generation_config.repetition_penalty = GENERATION_CONFIG["repetition_penalty"]
        model.generation_config.temperature = GENERATION_CONFIG["temperature"]
        model.generation_config.max_new_tokens = ENHANCED_CONFIG["max_completion_length"]
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        print("🔥 奖励增强生成配置完成")
    
    # 创建增强版奖励函数
    reward_functions = create_enhanced_reward_functions()
    print("✅ 奖励增强函数创建完成")
    
    reward_weights = [
        ENHANCED_SAMPLE_REWARDS_CONFIG['label_consistency_weight'],
        ENHANCED_SAMPLE_REWARDS_CONFIG['attribute_compliance_weight'],
        ENHANCED_SAMPLE_REWARDS_CONFIG['generation_quality_weight'],
        ENHANCED_BATCH_REWARDS_CONFIG['batch_diversity_weight'],
    ]
    
    print(f"📊 增强版权重: {reward_weights}")
    print(f"📊 权重总和: {sum(reward_weights)}")
    
    # LoRA配置
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM", inference_mode=False
    )
    
    # 配置GRPO训练参数
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
    
    # 初始化GRPO训练器
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    print("🔥 开始奖励增强GRPO训练...")
    print("📋 增强策略:")
    print("   - 🎯 放宽所有奖励的评分标准")
    print("   - 📈 提高奖励上限，给予更多激励")
    print("   - 🏆 增加基础奖励，减少零分情况")
    print("   - 🚀 优化训练参数，提高学习效率")
    print("   - 🎨 增加生成多样性和创造性")
    print("-" * 80)
    
    # 开始训练
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    training_time = (end_time - start_time) / 60
    print(f"⏱️ 训练耗时: {training_time:.2f}分钟")
    
    # 保存最终模型
    print("💾 保存训练后的模型...")
    trainer.save_model()
    
    # 绘制训练曲线
    print("📊 绘制奖励增强训练曲线...")
    plot_enhanced_training_curves(trainer, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("🎉 MNLI数据集奖励增强GRPO训练完成！")
    print(f"   训练时长: {training_time:.2f}分钟")
    print(f"   处理样本: {ENHANCED_CONFIG['max_train_samples']}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("   MNLI专用奖励特性:")
    print("     - ✅ 标签一致性：基于3个MNLI分类标签精确匹配")
    print("     - ✅ 属性符合度：8个MNLI属性约束检查")
    print("     - ✅ 生成质量：MNLI NLI特点和语言风格评估")
    print("     - ✅ 多样性奖励：批次内容多样性评估")
    print("     - ✅ 统一奖励上限为1.0，宽松评分标准促进学习")
    print("=" * 80)

if __name__ == "__main__":
    main()