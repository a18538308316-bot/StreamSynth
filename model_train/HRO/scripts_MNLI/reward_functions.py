import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import json
import re

try:
    from .batch_diversity_reward import initialize_batch_diversity_calculator
except ImportError:
    initialize_batch_diversity_calculator = None

from .attribute_handler import extract_attributes_from_input

# 全局变量
training_data_global = []
batch_size_global = 4
reward_calculator_global = None
novelsum_calculator_global = None
attr_loader_global = None
compliance_calculator_global = None
training_visualizer_global = None
optimized_sample_config_global = {}
optimized_batch_config_global = {}
embedding_model_path_global = None

# MNLI数据集奖励配置
SAMPLE_REWARDS_CONFIG = {
    "label_consistency_weight": 0.1,    # 标签一致性权重
    "attribute_compliance_weight": 0.1,     # 属性权重
    "generation_quality_weight": 0.5,       # 生成质量权重
}

BATCH_REWARDS_CONFIG = {
    "batch_diversity_weight": 0.3,          # 多样性权重
}

def initialize_reward_globals(
    training_data, 
    batch_size, 
    reward_calculator, 
    novelsum_calculator,
    attr_loader, 
    compliance_calculator,
    optimized_sample_config=None,
    optimized_batch_config=None,
    embedding_model_path=None
):
    """初始化奖励函数全局变量"""
    global training_data_global, batch_size_global, reward_calculator_global
    global novelsum_calculator_global, attr_loader_global, compliance_calculator_global
    global optimized_sample_config_global, optimized_batch_config_global, embedding_model_path_global
    
    training_data_global = training_data
    batch_size_global = batch_size
    reward_calculator_global = reward_calculator
    novelsum_calculator_global = novelsum_calculator
    attr_loader_global = attr_loader
    compliance_calculator_global = compliance_calculator
    optimized_sample_config_global = optimized_sample_config or SAMPLE_REWARDS_CONFIG
    optimized_batch_config_global = optimized_batch_config or BATCH_REWARDS_CONFIG
    embedding_model_path_global = embedding_model_path
    
    print("✅ MNLI奖励函数全局变量初始化完成")

    if embedding_model_path_global and initialize_batch_diversity_calculator:
        try:
            initialize_batch_diversity_calculator(embedding_model_path_global, device='cuda', k_penalty=2.0)
            print("✅ MNLI Batch多样性计算器初始化成功 (cuda)")
        except Exception as cuda_error:
            print(f"⚠️ MNLI Batch多样性计算器cuda初始化失败: {cuda_error}; 尝试使用CPU")
            try:
                initialize_batch_diversity_calculator(embedding_model_path_global, device='cpu', k_penalty=2.0)
                print("✅ MNLI Batch多样性计算器初始化成功 (cpu)")
            except Exception as cpu_error:
                print(f"⚠️ MNLI Batch多样性计算器初始化再次失败: {cpu_error}")

def set_training_visualizer(visualizer):
    """设置训练可视化器"""
    global training_visualizer_global
    training_visualizer_global = visualizer

def reward_label_consistency_batch(completions: List[str], **kwargs) -> List[float]:
    """MNLI数据集标签一致性奖励函数"""
    step = kwargs.get('step', 0)
    training_data = kwargs.get('training_data_global', training_data_global)
    
    if not training_data:
        return [0.5] * len(completions)
    
    rewards = []
    
    for i, completion in enumerate(completions):
        if i < len(training_data):
            sample = training_data[i]
            target_label = sample.get('label', 'neutral')
            
            # 从completion中提取标签
            extracted_label = 'neutral'  # 默认值
            
            if compliance_calculator_global is not None:
                extracted_label = compliance_calculator_global.extract_label_from_json(completion)
            else:
                # MNLI数据集：从completion中提取标签
                try:
                    import json
                    completion_json = json.loads(completion)
                    extracted_label = completion_json.get('output', 'neutral')
                except (json.JSONDecodeError, KeyError, TypeError):
                    # 如果JSON解析失败，使用关键词匹配
                    completion_lower = completion.lower()
                    if 'entailment' in completion_lower:
                        extracted_label = 'entailment'
                    elif 'contradiction' in completion_lower:
                        extracted_label = 'contradiction'
                    elif 'neutral' in completion_lower:
                        extracted_label = 'neutral'
                    else:
                        # 更复杂的关键词匹配
                        if any(word in completion_lower for word in ['true', 'correct', 'follows', 'implies']):
                            extracted_label = 'entailment'
                        elif any(word in completion_lower for word in ['false', 'wrong', 'conflicts', 'contradicts']):
                            extracted_label = 'contradiction'
                        else:
                            extracted_label = 'neutral'
            
            # 计算标签一致性分数
            if target_label.lower() == extracted_label.lower():
                score = 1.0  # 完全匹配
            else:
                score = 0.0  # 不匹配
            
            rewards.append(score)
            
            # 调试信息（仅前几个样本）
            if i < 2:
                print(f"   样本{i+1}: 目标标签={target_label}, 提取标签={extracted_label}, 分数={score:.2f}")
        else:
            rewards.append(0.5)  # 超出范围，给中等分数
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 从kwargs中移除step参数，避免重复传递
    if 'step' in kwargs:
        del kwargs['step']
    
    return rewards_tensor

def get_current_batch_attributes(**kwargs):
    """获取当前批次的属性信息"""
    training_data = kwargs.get('training_data_global', training_data_global)
    step = kwargs.get('step', 0)
    
    if not training_data:
        return {'target_label': 'neutral', 'length_premise': 50, 'length_hypothesis': 20}
    
    # 获取当前批次的属性
    batch_start = step * batch_size_global
    batch_end = min(batch_start + batch_size_global, len(training_data))
    
    if batch_start < len(training_data):
        current_batch = training_data[batch_start:batch_end]
        if current_batch:
            # 使用第一个样本的属性作为代表
            sample = current_batch[0]
            attributes = {
                'target_label': sample.get('label', 'neutral'),
                'length_premise': sample.get('length_premise', 50),
                'length_hypothesis': sample.get('length_hypothesis', 20)
            }
            print(f"   当前批次属性: 标签={attributes.get('target_label', '未知')}")
            return attributes
    
    return {'target_label': 'neutral', 'length_premise': 50, 'length_hypothesis': 20}

def extract_attributes_from_current_prompts(**kwargs):
    """从当前提示词中提取属性"""
    training_data = kwargs.get('training_data_global', training_data_global)
    step = kwargs.get('step', 0)
    
    if not training_data:
        return {}
    
    batch_start = step * batch_size_global
    batch_end = min(batch_start + batch_size_global, len(training_data))
    
    if batch_start < len(training_data):
        current_batch = training_data[batch_start:batch_end]
        if current_batch:
            # 使用第一个样本的输入作为代表
            sample = current_batch[0]
            input_text = sample.get('original_input', '')
            if input_text:
                attributes = extract_attributes_from_input(input_text)
                print(f"   从提示词提取属性: 标签={attributes.get('target_label', '未知')}")
                return attributes
    
    return {}

def reward_attribute_compliance_batch(completions: List[str], **kwargs) -> List[float]:
    """MNLI数据集属性符合度奖励函数"""
    step = kwargs.get('step', 0)
    training_data = kwargs.get('training_data_global', training_data_global)
    
    if not training_data:
        return [0.3] * len(completions)
    
    rewards = []
    
    for i, completion in enumerate(completions):
        if i < len(training_data):
            sample = training_data[i]
            sample_input = sample.get('original_input', '')
            
            # 提取MNLI属性
            attribute_constraints = extract_attributes_from_input(sample_input)
            
            # 获取目标标签
            target_label = sample.get('label', 'neutral')
            
            # 计算属性符合度分数
            attribute_score = 0.0
            total_attributes = 0
            
            # MNLI属性列表
            mnli_attributes = ['premise_domain', 'premise_style', 'hypothesis_transformation', 
                             'semantic_phenomenon', 'reasoning_type', 'distraction_type', 'label_strategy']
            
            for attr_name in mnli_attributes:
                if attr_name in attribute_constraints:
                    total_attributes += 1
                    target_value = attribute_constraints[attr_name]
                    
                    # 获取属性检查函数
                    if attr_loader_global:
                        check_func = attr_loader_global.get_attribute_check_function(
                            attr_name, target_value, label=target_label
                        )
                        attr_score = check_func(completion)
                        attribute_score += attr_score
            
            # 计算平均属性符合度
            if total_attributes > 0:
                avg_attribute_score = attribute_score / total_attributes
                rewards.append(avg_attribute_score)
            else:
                rewards.append(0.3)  # 没有属性约束，给基础分数
            
            # 调试信息
            if i < 2:
                print(f"   样本{i+1}: 目标标签={target_label}, 属性符合度={avg_attribute_score:.3f}")
        else:
            rewards.append(0.3)  # 超出范围，给基础分数
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 从kwargs中移除step参数，避免重复传递
    if 'step' in kwargs:
        del kwargs['step']
    
    return rewards_tensor

def reward_generation_quality_batch(completions: List[str], **kwargs) -> List[float]:
    """MNLI数据集生成质量奖励函数"""
    step = kwargs.get('step', 0)
    
    rewards = []
    
    for completion in completions:
        quality_score = 0.0
        
        # 检查长度
        word_count = len(completion.split())
        if word_count >= 50:  # MNLI通常较短
            quality_score += 0.3
        elif word_count >= 30:
            quality_score += 0.2
        elif word_count >= 15:
            quality_score += 0.1
        
        # 检查是否是有效的NLI内容
        if not completion.strip().startswith('{') and len(completion.strip()) > 5:
            quality_score += 0.4
            
            # 检查NLI的语言特征
            # 包含前提特征
            premise_words = ['premise', 'given', 'statement', 'fact']
            if any(word in completion.lower() for word in premise_words):
                quality_score += 0.1
            
            # 包含假设特征
            hypothesis_words = ['hypothesis', 'conclusion', 'inference', 'therefore']
            if any(word in completion.lower() for word in hypothesis_words):
                quality_score += 0.1
            
            # 包含逻辑词汇
            logic_words = ['because', 'therefore', 'however', 'although', 'if', 'then']
            if any(word in completion.lower() for word in logic_words):
                quality_score += 0.1
        
        rewards.append(min(quality_score, 1.0))
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 从kwargs中移除step参数，避免重复传递
    if 'step' in kwargs:
        del kwargs['step']
    
    return rewards_tensor

def extract_text_content_global(completion):
    """从completion中提取文本内容"""
    try:
        # 尝试解析JSON
        import json
        completion_json = json.loads(completion)
        if 'input' in completion_json:
            return completion_json['input']
        elif 'output' in completion_json:
            return completion_json['output']
        else:
            return completion
    except (json.JSONDecodeError, KeyError, TypeError):
        # 如果不是JSON，直接返回
        return completion