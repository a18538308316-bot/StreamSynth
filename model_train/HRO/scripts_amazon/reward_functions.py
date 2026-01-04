#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励函数模块 - 包含所有sample级别和batch级别的奖励函数
"""
import torch
import numpy as np
from datetime import datetime
try:
    from .dynamic_reward_scaler import set_dynamic_reward_config, tracer, get_scaled_weight, update_moving_average
    _DYNAMIC_AVAILABLE = True
except Exception as e:
    print(f"⚠️ 动态奖励模块不可用，使用静态奖励: {e}")
    _DYNAMIC_AVAILABLE = False
# 移除novelsum相关导入，但保留必要的工具函数

# 导入batch多样性奖励模块
try:
    from .batch_diversity_reward import reward_batch_diversity, initialize_batch_diversity_calculator
    print("✅ Batch多样性奖励模块导入成功")
except ImportError as e:
    print(f"⚠️ Batch多样性奖励模块导入失败: {e}")
    reward_batch_diversity = None
    initialize_batch_diversity_calculator = None

# 全局变量（将从主模块导入）
training_data_global = []
current_batch_index = 0
batch_size_global = 8
reward_calculator = None
novelsum_calculator = None
attr_loader = None
compliance_calculator = None
generation_history = []
reward_logs = []
current_training_step = 0
reward_call_counter = 0
current_prompt_attributes = {}
training_visualizer = None  # 训练可视化器

# Sample级别奖励权重（默认配置）
SAMPLE_REWARDS_CONFIG = {
    'sentiment_consistency_weight': 0.25,  # 情感标签一致性
    'attribute_compliance_weight': 0.15,   # 属性要求符合度
    'length_compliance_weight': 0.10,      # 长度要求符合度
}

# Batch级别奖励权重（默认配置）
BATCH_REWARDS_CONFIG = {
    'batch_diversity_weight': 0.30,  # 基于局部密度的batch多样性奖励
}

# 优化后的配置（将被动态更新）
CURRENT_SAMPLE_REWARDS_CONFIG = SAMPLE_REWARDS_CONFIG.copy()
CURRENT_BATCH_REWARDS_CONFIG = BATCH_REWARDS_CONFIG.copy()

# =============================================================================
# 工具函数（从novelsum模块迁移的必要函数）
# =============================================================================

def extract_text_content_global(text):
    """提取文本内容的全局函数（简化版本）"""
    if not text:
        return ""
    
    # 简单的文本清理
    text = str(text).strip()
    
    # 移除多余的空白字符
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text

def separate_prompt_and_generation_global(completion, prompt=""):
    """分离提示词和生成内容的全局函数（简化版本）"""
    if not completion:
        return "", ""
    
    completion = str(completion).strip()
    prompt = str(prompt).strip()
    
    # 如果提供了prompt，尝试从completion中移除它
    if prompt and completion.startswith(prompt):
        generation = completion[len(prompt):].strip()
        return prompt, generation
    
    # 如果没有prompt或无法分离，返回整个completion作为generation
    return prompt, completion

def get_current_batch_attributes(step, batch_size):
    """根据当前训练步数获取对应批次的属性信息"""
    global training_data_global, current_batch_index
    
    if not training_data_global:
        print("⚠️ 训练数据未初始化，使用默认属性")
        return {'target_sentiment': 'neutral', 'length': 200}
    
    # 使用循环访问，避免超出数据范围
    total_samples = len(training_data_global)
    
    # 使用模运算确保索引在有效范围内
    start_idx = (step * batch_size) % total_samples
    
    # 如果是循环访问，给出提示但不报错
    if step * batch_size >= total_samples:
        epoch_num = (step * batch_size) // total_samples + 1
        if step % 50 == 0:  # 每50步提示一次，避免日志过多
            print(f"🔄 训练进入第{epoch_num}轮，步数{step}，使用循环数据访问")
    
    # 获取当前批次的第一个样本的original_input
    try:
        sample = training_data_global[start_idx]
        if 'original_input' in sample:
            from scripts_amazon.attribute_handler import extract_attributes_from_input
            attributes = extract_attributes_from_input(sample['original_input'])
            print(f"🎯 Step {step}: 从数据中提取属性 - {attributes.get('target_sentiment', 'unknown')}")
            return attributes
        else:
            print(f"⚠️ 样本{start_idx}缺少original_input字段")
            return {'target_sentiment': 'neutral', 'cuisine': 'american', 'length': 200}
    except Exception as e:
        print(f"⚠️ 获取批次属性时出错: {e}")
        return {'target_sentiment': 'neutral', 'cuisine': 'american', 'length': 200}

def extract_attributes_from_current_prompts(prompts):
    """从当前的prompts中直接提取真实的目标属性"""
    if not prompts or len(prompts) == 0:
        return {}
    
    # 使用第一个prompt作为代表（同一batch的prompt应该有相同的要求）
    first_prompt = prompts[0]
    
    # 从prompt中提取属性
    from scripts_amazon.attribute_handler import extract_attributes_from_input
    
    # 直接从prompt文本中提取属性
    attributes = extract_attributes_from_input(first_prompt)
    
    return attributes

def update_current_prompt_attributes_from_prompts(prompts):
    """从当前的prompts更新属性（这是正确的方式）"""
    global current_prompt_attributes
    
    if prompts:
        current_prompt_attributes = extract_attributes_from_current_prompts(prompts)
        print(f"🎯 从prompt提取属性: 情感={current_prompt_attributes.get('target_sentiment', '未知')}")
    else:
        print("⚠️ 没有提供prompts，无法提取属性")

def update_current_prompt_attributes(step=None, batch_size=None):
    """更新当前批次的提示属性（从训练数据中提取或随机生成）- 已弃用，应使用从prompt提取的方式"""
    global current_prompt_attributes, current_batch_index, batch_size_global
    
    if step is not None and batch_size is not None:
        current_prompt_attributes = get_current_batch_attributes(step, batch_size)
        print(f"🔄 使用历史数据更新提示属性（可能不准确）: {current_prompt_attributes}")
    elif step is not None:
        current_prompt_attributes = get_current_batch_attributes(step, batch_size_global)
        print(f"🔄 使用默认batch_size更新提示属性")
    else:
        print("⚠️ 未提供step参数，无法更新提示属性")

def log_reward_details(step, reward_type, completions, rewards, **kwargs):
    global reward_logs, training_visualizer
    
    sample_details = []
    for i, (completion, reward) in enumerate(zip(completions[:3], rewards[:3])):
        sample_details.append({
            'completion_preview': completion[:100],
            'reward': float(reward),
            'completion_length': len(completion)
        })
    
    reward_log = {
        'step': step,
        'reward_type': reward_type,
        'mean_reward': float(torch.mean(rewards)),
        'std_reward': float(torch.std(rewards)),
        'min_reward': float(torch.min(rewards)),
        'max_reward': float(torch.max(rewards)),
        'num_samples': len(rewards),
        'sample_details': sample_details,
        'timestamp': datetime.now().isoformat()
    }
    
    reward_logs.append(reward_log)
    
    # 传递给训练可视化器（如果存在）
    if training_visualizer is not None and hasattr(training_visualizer, 'record_reward_data'):
        training_visualizer.record_reward_data(step, reward_type, rewards, completions)

# =============================================================================
# Sample级别奖励函数
# =============================================================================

def reward_sentiment_consistency_batch(completions, **kwargs):
    """情感标签一致性奖励"""
    global compliance_calculator, current_training_step, current_prompt_attributes
    rewards = []
    print(f"🎯 Step {current_training_step} - 评估情感标签一致性奖励 ({len(completions)}个样本)")
    
    # Debug: 查看kwargs内容
    if current_training_step <= 2:  # 前3步显示详细信息
        print(f"🔍 Debug kwargs keys: {list(kwargs.keys())}")
        if 'prompts' in kwargs:
            print(f"🔍 Prompts available: {len(kwargs['prompts'])}")
            
        # 显示完整的prompt和completion
        print("=" * 80)
        print(f"📝 STEP {current_training_step} - 完整的PROMPT和COMPLETION对比")
        print("=" * 80)
        
        for i, completion in enumerate(completions[:2]):  # 只显示前2个样本
            print(f"\n🔸 样本 {i+1}:")
            print(f"📥 完整PROMPT:")
            if 'prompts' in kwargs and i < len(kwargs['prompts']):
                prompt = kwargs['prompts'][i]
                print(f"'{prompt}'")
            else:
                print("  [PROMPT未找到]")
            
            print(f"\n📤 完整COMPLETION:")
            print(f"'{completion}'")
            
            print(f"\n🔍 分离后的生成内容:")
            try:
                if 'prompts' in kwargs and i < len(kwargs['prompts']):
                    prompt = kwargs['prompts'][i]
                    _, separated_generation = separate_prompt_and_generation_global(completion, prompt)
                else:
                    _, separated_generation = separate_prompt_and_generation_global(completion, "")
                print(f"'{separated_generation[:300]}...'")
            except Exception as e:
                print(f"   ❌ 分离失败: {e}")
                separated_generation = completion
            
            print("-" * 60)
        print("=" * 80)
        
    # 从prompt中提取真实的目标属性（正确方式）
    if 'prompts' in kwargs:
        update_current_prompt_attributes_from_prompts(kwargs['prompts'])
    else:
        # 如果没有prompts，尝试旧方式（但不太准确）
        if not current_prompt_attributes:
            update_current_prompt_attributes()
    
    target_sentiment = current_prompt_attributes.get('target_sentiment', 'neutral')
    
    for i, completion in enumerate(completions):
        # 检查compliance_calculator是否为None（Amazon数据集情况）
        if compliance_calculator is None:
            # 使用简单的关键词匹配作为兜底
            completion_lower = completion.lower()
            if target_sentiment.lower() in completion_lower:
                score = 1.0
            else:
                # 检查情感相关词汇
                sentiment_words = {
                    'very positive': ['excellent', 'amazing', 'fantastic', 'outstanding', 'perfect', 'love', 'best'],
                    'positive': ['good', 'great', 'nice', 'satisfied', 'happy', 'recommend', 'works well'],
                    'neutral': ['okay', 'average', 'fine', 'decent', 'acceptable'],
                    'negative': ['bad', 'poor', 'disappointed', 'issues', 'problems', 'not good'],
                    'very negative': ['terrible', 'awful', 'horrible', 'worst', 'hate', 'waste', 'broken']
                }
                
                words = sentiment_words.get(target_sentiment.lower(), [])
                matches = sum(1 for word in words if word in completion_lower)
                score = min(matches / max(len(words), 1), 1.0)
        else:
            score = compliance_calculator.calculate_sentiment_consistency(completion, target_sentiment)
        
        # 正向激励策略：完全匹配给高奖励，不匹配给中性奖励（避免负分）
        raw_norm = max(0.0, min(1.0, score))
        dyn_w = get_scaled_weight("sentiment", current_training_step) if _DYNAMIC_AVAILABLE else 0.5
        continuous_reward = (raw_norm ** 2) * dyn_w
        if score >= 1.0:
            legacy_reward = 0.5
        elif score >= 0.5:
            legacy_reward = 0.2
        else:
            legacy_reward = 0.0
        final_reward = max(legacy_reward, continuous_reward)
        ma = update_moving_average("sentiment_raw", raw_norm) if _DYNAMIC_AVAILABLE else raw_norm
        rewards.append(final_reward)
        
        if i < 2:
            if compliance_calculator is not None:
                extracted_sentiment = compliance_calculator.extract_sentiment_from_json(completion)
            else:
                # Amazon数据集：从completion中提取情感标签
                try:
                    import json
                    completion_json = json.loads(completion)
                    extracted_sentiment = completion_json.get('output', 'unknown')
                except (json.JSONDecodeError, KeyError, TypeError):
                    # 如果JSON解析失败，使用关键词匹配
                    completion_lower = completion.lower()
                    if any(word in completion_lower for word in ['excellent', 'amazing', 'fantastic', 'outstanding', 'perfect', 'love', 'best']):
                        extracted_sentiment = 'very positive'
                    elif any(word in completion_lower for word in ['good', 'great', 'nice', 'satisfied', 'happy', 'recommend']):
                        extracted_sentiment = 'positive'
                    elif any(word in completion_lower for word in ['okay', 'average', 'fine', 'decent', 'acceptable']):
                        extracted_sentiment = 'neutral'
                    elif any(word in completion_lower for word in ['bad', 'poor', 'disappointed', 'issues', 'problems']):
                        extracted_sentiment = 'negative'
                    elif any(word in completion_lower for word in ['terrible', 'awful', 'horrible', 'worst', 'hate', 'waste', 'broken']):
                        extracted_sentiment = 'very negative'
                    else:
                        extracted_sentiment = 'unknown'
            # 显示处理后的生成内容而不是原始completion
            generated_text = extract_text_content_global(completion)
            print(f"   样本{i+1}: 目标情感={target_sentiment}, 提取情感={extracted_sentiment}, raw={raw_norm:.2f}, 连续={continuous_reward:.4f}, 最终={final_reward:.4f}, MA={ma:.3f}")
        # tracer 在 dynamic_reward_scaler 中可能是一个属性对象而非可调用函数
        if _DYNAMIC_AVAILABLE and tracer is not None and hasattr(tracer, "log"):
            tracer.log(current_training_step, task="amazon", component="sentiment", raw_score=raw_norm, final_reward=final_reward,
                         extra={"legacy_reward": legacy_reward, "dyn_weight": dyn_w})
            print(f"   生成内容: {generated_text[:100]}...")
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 从kwargs中移除step参数，避免重复传递
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('step', None)
    
    log_reward_details(current_training_step, "sentiment_consistency", completions, rewards_tensor, **kwargs_copy)
    return rewards_tensor

def reward_attribute_compliance_batch(completions, **kwargs):
    """属性要求符合度奖励"""
    global compliance_calculator, current_training_step, current_prompt_attributes
    rewards = []
    print(f"🔍 Step {current_training_step} - 评估属性符合度奖励 ({len(completions)}个样本)")
    
    # 确保有当前提示属性
    if not current_prompt_attributes:
        update_current_prompt_attributes()
    
    for i, completion in enumerate(completions):
        total_score = 0.0
        attribute_count = 0
        
        # 评估菜系属性
        if 'cuisine' in current_prompt_attributes:
            cuisine_score = compliance_calculator.calculate_cuisine_compliance(
                completion, current_prompt_attributes['cuisine']
            )
            total_score += cuisine_score
            attribute_count += 1
        
        # 评估其他属性
        for attr_name in ['style', 'price_range', 'service_quality', 'atmosphere']:
            if attr_name in current_prompt_attributes:
                attr_score = compliance_calculator.calculate_attribute_keyword_match(
                    completion, 
                    attr_name, 
                    current_prompt_attributes[attr_name],
                    current_prompt_attributes.get('target_sentiment')
                )
                total_score += attr_score * 0.5  # 降低其他属性的权重
                attribute_count += 0.5
        
        # 计算平均分数
        if attribute_count > 0:
            avg_score = total_score / attribute_count
        else:
            avg_score = 0.5
        
        # 正向激励策略：符合属性给奖励，不符合不惩罚
        raw_norm = max(0.0, min(1.0, avg_score))
        dyn_w = get_scaled_weight("attribute", current_training_step) if _DYNAMIC_AVAILABLE else 0.25
        continuous_reward = (raw_norm ** 1.5) * dyn_w
        if avg_score >= 0.8:
            legacy_reward = 0.25
        elif avg_score >= 0.6:
            legacy_reward = 0.15
        elif avg_score >= 0.4:
            legacy_reward = 0.1
        elif avg_score >= 0.2:
            legacy_reward = 0.05
        else:
            legacy_reward = 0.0
        final_reward = max(legacy_reward, continuous_reward)
        ma = update_moving_average("attribute_raw", raw_norm) if _DYNAMIC_AVAILABLE else raw_norm
        rewards.append(final_reward)
        
        if i < 2:
            cuisine = current_prompt_attributes.get('cuisine', 'N/A')
            print(f"   样本{i+1}: 目标菜系={cuisine}, raw={raw_norm:.2f}, 连续={continuous_reward:.4f}, 最终={final_reward:.4f}, MA={ma:.3f}")
        if _DYNAMIC_AVAILABLE and tracer is not None and hasattr(tracer, "log"):
            tracer.log(current_training_step, task="amazon", component="attribute", raw_score=raw_norm, final_reward=final_reward,
                         extra={"legacy_reward": legacy_reward, "dyn_weight": dyn_w})
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 从kwargs中移除step参数，避免重复传递
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('step', None)
    
    log_reward_details(current_training_step, "attribute_compliance", completions, rewards_tensor, **kwargs_copy)
    return rewards_tensor

def reward_length_compliance_batch(completions, **kwargs):
    """长度要求符合度奖励"""
    global compliance_calculator, current_training_step, current_prompt_attributes
    rewards = []
    print(f"📏 Step {current_training_step} - 评估长度符合度奖励 ({len(completions)}个样本)")
    
    # 确保有当前提示属性
    if not current_prompt_attributes:
        update_current_prompt_attributes()
    
    target_length = current_prompt_attributes.get('length', 200)
    
    for i, completion in enumerate(completions):
        score = compliance_calculator.calculate_length_compliance(completion, target_length, tolerance=25)
        
        # 正向激励策略：在范围内给奖励，超出范围不惩罚
        raw_norm = max(0.0, min(1.0, score))
        dyn_w = get_scaled_weight("length", current_training_step) if _DYNAMIC_AVAILABLE else 0.2
        continuous_reward = (1 / (1 + np.exp(-8 * (raw_norm - 0.5)))) * dyn_w
        if score >= 1.0:
            legacy_reward = 0.2
        elif score >= 0.8:
            legacy_reward = 0.1
        elif score >= 0.5:
            legacy_reward = 0.05
        else:
            legacy_reward = 0.0
        final_reward = max(legacy_reward, continuous_reward)
        ma = update_moving_average("length_raw", raw_norm) if _DYNAMIC_AVAILABLE else raw_norm
        rewards.append(final_reward)
        
        if i < 2:
            # 计算实际长度用于显示
            text_content = extract_text_content_global(completion)
            actual_length = len(text_content.split())
            
            # 显示目标长度信息
            if isinstance(target_length, dict):
                length_info = f"目标范围={target_length['min']}-{target_length['max']}"
            else:
                length_info = f"目标长度={target_length}"
            
            print(f"   样本{i+1}: {length_info}, raw={raw_norm:.2f}, 连续={continuous_reward:.4f}, 最终={final_reward:.4f}, MA={ma:.3f}")
        if _DYNAMIC_AVAILABLE and tracer is not None and hasattr(tracer, "log"):
            tracer.log(current_training_step, task="amazon", component="length", raw_score=raw_norm, final_reward=final_reward,
                         extra={"legacy_reward": legacy_reward, "dyn_weight": dyn_w})
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 从kwargs中移除step参数，避免重复传递
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('step', None)
    
    log_reward_details(current_training_step, "length_compliance", completions, rewards_tensor, **kwargs_copy)
    return rewards_tensor

# =============================================================================
# 生成质量奖励函数
# =============================================================================

def extract_clean_review_global(generated_text, prompt=""):
    """从生成文本中提取干净的评论内容（保持JSON格式）"""
    import re
    import json
    
    # 移除prompt部分
    if prompt and generated_text.startswith(prompt):
        content = generated_text[len(prompt):].strip()
    else:
        content = generated_text.strip()
    
    # 尝试提取完整的JSON
    json_pattern = r'\{[^{}]*"input"\s*:\s*"[^"]*"[^{}]*"output"\s*:\s*"[^"]*"[^{}]*\}'
    json_match = re.search(json_pattern, content)
    if json_match:
        try:
            # 验证JSON格式是否正确
            json_str = json_match.group(0)
            parsed_json = json.loads(json_str)
            if 'input' in parsed_json and 'output' in parsed_json:
                return json_str  # 返回完整的JSON
        except json.JSONDecodeError:
            pass
    
    # 如果没找到完整JSON，尝试提取input字段的内容
    input_match = re.search(r'"input"\s*:\s*"([^"]*)"', content)
    if input_match:
        input_content = input_match.group(1)
        # 尝试提取output字段
        output_match = re.search(r'"output"\s*:\s*"([^"]*)"', content)
        if output_match:
            output_content = output_match.group(1)
            # 重构JSON
            return f'{{"input": "{input_content}", "output": "{output_content}"}}'
        else:
            # 只有input，添加默认output
            return f'{{"input": "{input_content}", "output": "unknown"}}'
    
    # 如果都没找到，尝试提取引号内的长内容
    quote_matches = re.findall(r'"([^"]{20,})"', content)
    if quote_matches:
        # 选择最长的匹配作为主要内容，包装成JSON
        longest_content = max(quote_matches, key=len)
        return f'{{"input": "Text: {longest_content}", "output": "unknown"}}'
    
    # 最后的兜底：移除指令性文本，包装成JSON
    clean_patterns = [
        r'Here is an example.*?:',
        r'Here\'s an example.*?:',
        r'Do NOT.*?\.',
        r'Note that.*?\.',
        r'```.*?```',
        r'""".*?"""',
        r'Example.*?:',
    ]
    
    for pattern in clean_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理多余空白和特殊字符
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'^["\'\s]+|["\'\s]+$', '', content)
    
    if len(content) > 10:
        return f'{{"input": "Text: {content}", "output": "unknown"}}'
    else:
        return '{"input": "Text: Invalid generation", "output": "unknown"}'

def calculate_generation_quality_score_global(generated_text, prompt=""):
    """生成质量评分（减轻过度扣分，保持JSON结构主路径加分）"""
    import json
    import re

    clean_content = extract_clean_review_global(generated_text, prompt)

    # 提高起始基线，防止一次扣分直接归零
    quality_score = 0.2

    # 编程/模板类不良模式（轻量化惩罚）
    bad_patterns = [
        'import ', 'def ', 'class ', 'function', '```', 'python', 'code', 'script',
        'return ', 'print(', 'if __name__', 'from ', 'pipeline', 'random.',
        'tokenizer', 'transformers', 'torch', 'numpy'
    ]
    bad_count = sum(1 for pattern in bad_patterns if pattern.lower() in generated_text.lower())
    if bad_count > 0:
        quality_score -= min(0.6, bad_count * 0.2)  # 上限降低

    # 重复字符惩罚减轻
    repeat_matches = re.findall(r'(.)\1{10,}', generated_text)
    if repeat_matches:
        quality_score -= 0.3

    # 指令性短语扣分减轻
    instruction_phrases = [
        'here is an example', 'here\'s an example', 'do not add', 'note that you',
        'template', 'format', 'please see', 'answer:', 'step 1:', 'step 2:',
        'feel free', 'let me know', 'best regards'
    ]
    bad_instructions = sum(1 for phrase in instruction_phrases if phrase.lower() in generated_text.lower())
    if bad_instructions > 0:
        quality_score -= min(0.3, bad_instructions * 0.1)

    # 奖励稳定 JSON 结构
    try:
        parsed_json = json.loads(clean_content)
        if 'input' in parsed_json and 'output' in parsed_json:
            quality_score += 0.4
            input_content = parsed_json.get('input', '')
            if input_content.startswith('Text: ') and len(input_content) > 20:
                quality_score += 0.3
                review_text = input_content[6:]
                word_count = len(review_text.split())
                # 长度奖励改为更宽容：过短仍扣分，过长轻微扣分
                if word_count < 10:
                    quality_score -= 0.2
                elif word_count > 500:
                    quality_score -= 0.05
                else:
                    # 平滑长度奖励：20~300 接近满分，其余逐渐下降
                    if word_count < 20:
                        length_bonus = (word_count / 20) * 0.2
                    elif word_count <= 300:
                        length_bonus = 0.2
                    else:
                        # 300~500 线性衰减到 0.05
                        length_bonus = max(0.05, 0.2 - (word_count - 300) / 200 * 0.15)
                    quality_score += length_bonus
            else:
                quality_score -= 0.05  # 减轻 input 格式惩罚

            output_content = parsed_json.get('output', '')
            valid_sentiments = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
            if output_content in valid_sentiments:
                quality_score += 0.1
            elif output_content == 'unknown':
                pass  # 不再扣分
            else:
                quality_score -= 0.1
        else:
            quality_score -= 0.15  # 缺少字段惩罚减轻
    except json.JSONDecodeError:
        quality_score -= 0.25  # JSON失败惩罚减轻

    # 噪音/格式化符号惩罚减轻
    noise_patterns = [r'```', r'"""', r'\{[^}]*\}[^}]*\{', r'\\n\\n\\n+']
    noise_hits = 0
    for pattern in noise_patterns:
        if re.search(pattern, generated_text):
            noise_hits += 1
    if noise_hits:
        quality_score -= min(0.15, noise_hits * 0.05)

    # 移除重复 code_indicators 二次惩罚（已在 bad_patterns 覆盖）

    return max(0.0, min(1.0, quality_score))

def reward_generation_quality_batch(completions, **kwargs):
    """生成质量奖励函数"""
    global current_training_step
    rewards = []
    prompts = kwargs.get('prompts', [''] * len(completions))
    
    print(f"🎨 Step {current_training_step} - 评估生成质量奖励 ({len(completions)}个样本)")
    
    for i, completion in enumerate(completions):
        prompt = prompts[i] if i < len(prompts) else ""
        quality_score = calculate_generation_quality_score_global(completion, prompt)
        
        # 正向激励策略：高质量给奖励，低质量不惩罚
        raw_norm = max(0.0, min(1.0, quality_score))
        dyn_w = get_scaled_weight("generation_quality", current_training_step) if _DYNAMIC_AVAILABLE else 0.3
        continuous_reward = (raw_norm ** 1.2) * dyn_w
        if quality_score >= 0.9:
            legacy_reward = 0.3
        elif quality_score >= 0.7:
            legacy_reward = 0.2
        elif quality_score >= 0.5:
            legacy_reward = 0.1
        elif quality_score >= 0.3:
            legacy_reward = 0.05
        else:
            legacy_reward = 0.0
        reward = max(legacy_reward, continuous_reward)
        ma = update_moving_average("generation_quality_raw", raw_norm) if _DYNAMIC_AVAILABLE else raw_norm
        rewards.append(reward)
        
        if i < 2:  # 显示前2个样本的详细信息
            clean_content = extract_clean_review_global(completion, prompt)
            print(f"   样本{i+1}: raw质量={raw_norm:.2f}, 连续={continuous_reward:.4f}, 最终={reward:.4f}, MA={ma:.3f}")
        if _DYNAMIC_AVAILABLE and tracer is not None and hasattr(tracer, "log"):
            tracer.log(current_training_step, task="amazon", component="generation_quality", raw_score=raw_norm, final_reward=reward,
                         extra={"legacy_reward": legacy_reward, "dyn_weight": dyn_w})
            print(f"   清理后内容: {clean_content[:100]}...")
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 从kwargs中移除step参数，避免重复传递
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('step', None)
    
    log_reward_details(current_training_step, "generation_quality", completions, rewards_tensor, **kwargs_copy)
    return rewards_tensor

# =============================================================================
# Batch级别奖励函数
# =============================================================================

def reward_yelp_semantic_diversity_batch(completions, **kwargs):
    """餐厅评论语义多样性奖励（结合NovelSum思想和Yelp数据特色，支持动态参考数据）"""
    global novelsum_calculator, current_training_step
    
    print(f"🍽️ Step {current_training_step} - 评估餐厅评论语义多样性奖励 (batch size: {len(completions)})")
    
    try:
        # 从completions中提取实际文本内容
        texts = [extract_text_content_global(completion) for completion in completions]
        texts = [text for text in texts if text and len(text) > 10]
        
        if len(texts) < 2:
            print("   ⚠️ 有效文本少于2个，返回基础奖励")
            return torch.zeros(len(completions), dtype=torch.float32)
        
        # 使用NovelSum计算多样性
        novelsum_score = novelsum_calculator.calculate_novelsum_score(
            texts,
            density_power=0.5,
            distance_power=1.0,
            neighbors=min(10, len(texts))
        )
        
        # 计算餐厅评论特色多样性
        print(f"🔍 DEBUG: 开始计算餐厅多样性...")
        restaurant_diversity = calculate_restaurant_specific_diversity(texts)
        print(f"🔍 DEBUG: 餐厅多样性计算完成: {restaurant_diversity} (类型: {type(restaurant_diversity)})")
        
        # 结合两种多样性度量
        print(f"🔍 DEBUG: 开始结合多样性度量...")
        print(f"🔍 DEBUG: novelsum_score = {novelsum_score} (类型: {type(novelsum_score)})")
        print(f"🔍 DEBUG: restaurant_diversity = {restaurant_diversity} (类型: {type(restaurant_diversity)})")
        
        combined_diversity = (novelsum_score * 0.6 + restaurant_diversity * 0.4)
        print(f"🔍 DEBUG: 初始combined_diversity = {combined_diversity} (类型: {type(combined_diversity)})")
        
        combined_diversity = float(combined_diversity)  # 确保是标量
        print(f"🔍 DEBUG: 转换后combined_diversity = {combined_diversity} (类型: {type(combined_diversity)})")
        
        # 将batch级别的奖励分配给每个样本
        print(f"🔍 DEBUG: 开始计算batch奖励...")
        batch_reward = (combined_diversity - 0.5) * 0.6  # 标准化到[-0.3, 0.3]
        print(f"🔍 DEBUG: 初始batch_reward = {batch_reward} (类型: {type(batch_reward)})")
        
        batch_reward = float(batch_reward)  # 确保是标量
        print(f"🔍 DEBUG: 转换后batch_reward = {batch_reward} (类型: {type(batch_reward)})")
        
        rewards = torch.full((len(completions),), batch_reward, dtype=torch.float32)
        print(f"🔍 DEBUG: rewards张量创建完成: shape={rewards.shape}")
        
        # 动态更新参考数据池（更宽松的条件以促进学习）
        print(f"🔍 DEBUG: 检查是否更新参考池...")
        print(f"🔍 DEBUG: hasattr(novelsum_calculator, 'add_training_samples') = {hasattr(novelsum_calculator, 'add_training_samples')}")
        print(f"🔍 DEBUG: batch_reward = {batch_reward}, batch_reward > -0.1 = {batch_reward > -0.1}")
        
        if hasattr(novelsum_calculator, 'add_training_samples') and batch_reward > -0.1:
            try:
                # 计算每个样本的质量分数（基于多样性和长度）
                quality_scores = []
                for text in texts:
                    length_score = min(1.0, len(text.split()) / 150.0)  # 长度归一化
                    
                    # 安全地计算individual_novelty
                    if hasattr(novelsum_calculator, 'reference_manager'):
                        individual_novelty = novelsum_calculator.reference_manager.compute_novelty_score(text)
                        # 确保是标量
                        if isinstance(individual_novelty, torch.Tensor):
                            individual_novelty = float(individual_novelty.item())
                        else:
                            individual_novelty = float(individual_novelty)
                    else:
                        individual_novelty = 0.5
                    
                    quality = (individual_novelty * 0.7 + length_score * 0.3)
                    quality_scores.append(float(quality))
                
                # 添加高质量样本到参考池
                added_count = novelsum_calculator.add_training_samples(texts, quality_scores)
                if added_count > 0:
                    print(f"   📊 添加{added_count}个高质量样本到动态参考池")
                    
            except Exception as e:
                print(f"   ⚠️ 动态参考池更新失败: {e}")
                import traceback
                print(f"   详细错误: {traceback.format_exc()}")
        
        
        print(f"   NovelSum分数: {novelsum_score:.3f}, 餐厅多样性: {restaurant_diversity:.3f}")
        print(f"   综合多样性: {combined_diversity:.3f}, batch奖励: {batch_reward:.4f}")
        
        # 定期打印参考池统计信息
        if current_training_step % 20 == 0 and hasattr(novelsum_calculator, 'get_reference_statistics'):
            stats = novelsum_calculator.get_reference_statistics()
            print(f"   📈 参考池状态: {stats.get('total_count', 0)}样本 (原始:{stats.get('original_count', 0)}, 动态:{stats.get('dynamic_count', 0)})")
        
        # 从kwargs中移除step参数，避免重复传递
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('step', None)
        log_reward_details(current_training_step, "yelp_semantic_diversity", completions, rewards, **kwargs_copy)
        return rewards
        
    except Exception as e:
        print(f"   ⚠️ 多样性计算失败: {e}")
        return torch.zeros(len(completions), dtype=torch.float32)

def reward_inter_sample_diversity_batch(completions, **kwargs):
    """批次内样本多样性奖励"""
    global current_training_step
    
    print(f"🔄 Step {current_training_step} - 评估批次内多样性奖励 (batch size: {len(completions)})")
    
    try:
        texts = [extract_text_content_global(completion) for completion in completions]
        texts = [text for text in texts if text and len(text) > 10]
        
        if len(texts) < 2:
            return torch.zeros(len(completions), dtype=torch.float32)
        
        # 使用简单的jaccard距离计算多样性
        def jaccard_similarity(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        # 计算所有样本对的多样性
        diversities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = jaccard_similarity(texts[i], texts[j])
                diversity = 1.0 - similarity
                diversities.append(diversity)
        
        if not diversities:
            return torch.zeros(len(completions), dtype=torch.float32)
        
        avg_diversity = np.mean(diversities)
        avg_diversity = float(avg_diversity)  # 确保是标量
        
        # 下面的计算应当仍在try块中，否则try后将缺少except导致语法错误
        raw_norm = max(0.0, min(1.0, avg_diversity))
        dyn_w = get_scaled_weight("diversity", current_training_step) if _DYNAMIC_AVAILABLE else 0.3
        continuous_reward = (raw_norm ** 1.3) * dyn_w
        legacy_batch_reward = (avg_diversity - 0.5) * 0.4  # 原始映射
        batch_reward = max(legacy_batch_reward, continuous_reward)
        ma = update_moving_average("diversity_pair_raw", raw_norm) if _DYNAMIC_AVAILABLE else raw_norm
        rewards = torch.full((len(completions),), float(batch_reward), dtype=torch.float32)
        
        print(f"   平均Jaccard多样性: {avg_diversity:.3f}, raw={raw_norm:.3f}, 连续={continuous_reward:.4f}, 最终batch奖励={batch_reward:.4f}, MA={ma:.3f}")
        if _DYNAMIC_AVAILABLE and tracer is not None and hasattr(tracer, "log"):
            tracer.log(current_training_step, task="amazon", component="inter_sample_diversity", raw_score=raw_norm, final_reward=float(batch_reward),
                         extra={"legacy_reward": legacy_batch_reward, "dyn_weight": dyn_w})
        
        # 从kwargs中移除step参数，避免重复传递
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('step', None)
        log_reward_details(current_training_step, "inter_sample_diversity", completions, rewards, **kwargs_copy)
        return rewards
        
    except Exception as e:
        print(f"   ⚠️ 批次多样性计算失败: {e}")
        return torch.zeros(len(completions), dtype=torch.float32)

# =============================================================================
# 综合奖励函数
# =============================================================================

def create_weighted_reward_functions():
    """创建加权的奖励函数 - 包含Sample级别和Batch级别的奖励"""
    
    def weighted_sentiment_consistency(completions, **kwargs):
        """加权情感一致性奖励函数"""
        global reward_call_counter, current_training_step
        
        # 更新全局计数器和训练步数
        reward_call_counter += 1
        current_training_step = reward_call_counter // 5  # 每5次奖励函数调用为一个训练步（现在有5个奖励函数）
        
        # 更新当前批次的提示属性
        update_current_prompt_attributes(current_training_step)
        
        base_rewards = reward_sentiment_consistency_batch(completions, **kwargs)
        return base_rewards  # 返回未加权的基础奖励
    
    def weighted_attribute_compliance(completions, **kwargs):
        """属性符合度奖励函数（返回未加权值）"""
        base_rewards = reward_attribute_compliance_batch(completions, **kwargs)
        return base_rewards  # 返回未加权的基础奖励
    
    def weighted_length_compliance(completions, **kwargs):
        """长度符合度奖励函数（返回未加权值）"""
        base_rewards = reward_length_compliance_batch(completions, **kwargs)
        return base_rewards  # 返回未加权的基础奖励
    
    def weighted_generation_quality(completions, **kwargs):
        """生成质量奖励函数"""
        base_rewards = reward_generation_quality_batch(completions, **kwargs)
        return base_rewards  # 返回未加权的基础奖励
    
    def weighted_batch_diversity(completions, **kwargs):
        """Batch级别多样性奖励函数"""
        if reward_batch_diversity is None:
            print("⚠️ Batch多样性奖励函数不可用，返回中性奖励")
            return [0.0] * len(completions)
        
        base_rewards = reward_batch_diversity(completions, **kwargs)
        return base_rewards  # 返回未加权的基础奖励
    
    # 设置函数名
    weighted_sentiment_consistency.__name__ = "reward_sentiment_consistency"
    weighted_attribute_compliance.__name__ = "reward_attribute_compliance"
    weighted_length_compliance.__name__ = "reward_length_compliance"
    weighted_generation_quality.__name__ = "reward_generation_quality"
    weighted_batch_diversity.__name__ = "reward_batch_diversity"
    
    return [
        weighted_sentiment_consistency,    # Sample级别: 情感标签一致性
        weighted_attribute_compliance,     # Sample级别: 属性要求符合度  
        weighted_length_compliance,        # Sample级别: 长度要求符合度
        weighted_generation_quality,       # Sample级别: 生成质量控制
        weighted_batch_diversity,          # Batch级别: 基于局部密度的多样性奖励
    ]

def save_training_logs(output_dir):
    """保存训练日志"""
    global reward_logs
    
    if reward_logs:
        logs_file = f"{output_dir}/reward_logs.json"
        import json
        with open(logs_file, 'w', encoding='utf-8') as f:
            json.dump(reward_logs, f, indent=2, ensure_ascii=False)
        print(f"📊 奖励日志已保存: {logs_file}")

def initialize_reward_globals(td_global, bs_global, rc, nc, al, cc, optimized_sample_config=None, optimized_batch_config=None, embedding_model_path=None):
    """初始化奖励函数模块的全局变量（Sample + Batch级别）"""
    global training_data_global, batch_size_global, reward_calculator, novelsum_calculator, attr_loader, compliance_calculator
    global CURRENT_SAMPLE_REWARDS_CONFIG, CURRENT_BATCH_REWARDS_CONFIG
    
    training_data_global = td_global
    batch_size_global = bs_global
    reward_calculator = rc
    novelsum_calculator = nc  # 保留兼容性，但不使用
    attr_loader = al
    compliance_calculator = cc
    
    # 更新配置（如果提供了优化配置）
    if optimized_sample_config:
        CURRENT_SAMPLE_REWARDS_CONFIG.update(optimized_sample_config)
        print(f"✅ 使用优化的Sample奖励配置: {CURRENT_SAMPLE_REWARDS_CONFIG}")
    
    # 更新batch配置
    if optimized_batch_config:
        CURRENT_BATCH_REWARDS_CONFIG.update(optimized_batch_config)
        print(f"✅ 使用优化的Batch奖励配置: {CURRENT_BATCH_REWARDS_CONFIG}")
    
    # 初始化batch多样性计算器
    if embedding_model_path and initialize_batch_diversity_calculator:
        try:
            initialize_batch_diversity_calculator(embedding_model_path, device='cuda', k_penalty=2.0)
            print("✅ Batch多样性计算器初始化成功")
        except Exception as e:
            print(f"⚠️ Batch多样性计算器初始化失败: {e}")
    
    print("✅ 奖励函数模块全局变量初始化完成（Sample + Batch级别）")

def set_training_visualizer(visualizer):
    """设置训练可视化器"""
    global training_visualizer
    training_visualizer = visualizer
    print(f"✅ 训练可视化器已设置")