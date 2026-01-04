#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块 - 数据加载、格式转换和预处理
"""
import json
import os
from datasets import Dataset
from scripts_amazon.attribute_handler import extract_attributes_from_input

def load_synthesis_data(file_path, max_samples=None):
    """加载合成数据，转换为GRPO兼容格式"""
    print(f"Loading synthesis data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    formatted_data = []
    for item in data:
        # 解析requirements部分以获取真实的目标属性
        if 'instruction' in item and 'input' in item and 'output' in item:
            try:
                # 从output中解析JSON获取真实的情感标签（Amazon数据格式）
                output_json = json.loads(item['output'])
                true_target_sentiment = output_json.get('output', 'neutral')  # 这是真实的目标情感
                review_text = output_json.get('input', '').replace('Text: ', '')
                generated_sentiment = true_target_sentiment  # 在Amazon数据中，output就是目标情感
                
                # 如果output解析失败，从input中提取作为兜底
                if true_target_sentiment == 'neutral':
                    requirements_text = item['input']
                    attributes = extract_attributes_from_input(requirements_text)
                    true_target_sentiment = attributes.get('target_sentiment', 'neutral')
                
                # 构建GRPO格式的messages（只需要prompt，不需要参考答案）
                user_message = f"{item['instruction']}\n\n{item['input']}"
                
                formatted_item = {
                    'messages': [
                        {'role': 'user', 'content': user_message}
                        # 注意：GRPO不需要assistant消息，模型会自己生成
                    ],
                    # 关键信息：使用requirements中的真实目标
                    'sentiment': true_target_sentiment,  # 真实目标情感
                    'generated_sentiment': generated_sentiment,  # GPT生成的情感（供对比）
                    'review_text': review_text,  # 生成的评论文本
                    'original_input': item['input']  # 保存原始requirements用于属性提取
                }
                formatted_data.append(formatted_item)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ 跳过格式错误的样本: {e}")
                continue
                
        # 兼容原有的messages格式
        elif 'messages' in item and len(item['messages']) >= 1:
            # 对于已有的messages格式，提取真实目标情感
            user_content = item['messages'][0].get('content', '')
            attributes = extract_attributes_from_input(user_content)
            
            formatted_item = {
                'messages': item['messages'],
                'sentiment': attributes.get('target_sentiment', 'neutral'),
                'original_input': user_content
            }
            
            # 如果有其他字段，也保留
            for key in ['review_text', 'generated_sentiment']:
                if key in item:
                    formatted_item[key] = item[key]
                    
            formatted_data.append(formatted_item)
        else:
            print(f"⚠️ 跳过不支持的数据格式: {list(item.keys())}")
    
    print(f"Loaded {len(formatted_data)} samples")
    return formatted_data

def create_sentiment_grouped_dataset(data, batch_size=4):
    """创建按情感标签分组的数据集，优化小batch训练"""
    print(f"🔄 按情感标签重组数据集 (目标batch大小: {batch_size})...")
    
    # 按情感标签分组
    sentiment_groups = {
        'very negative': [],
        'negative': [],
        'neutral': [],
        'positive': [],
        'very positive': []
    }
    
    for item in data:
        sentiment = item.get('sentiment', 'neutral')
        if sentiment in sentiment_groups:
            sentiment_groups[sentiment].append(item)
        else:
            # 处理未知情感标签
            print(f"⚠️ 未知情感标签: {sentiment}，归类为neutral")
            sentiment_groups['neutral'].append(item)
                
    
    # 打印分组统计
    print("📊 情感标签分组统计:")
    for sentiment, items in sentiment_groups.items():
        print(f"   {sentiment}: {len(items)}个样本")
    
    # 重新组织数据，确保同一batch内尽可能是同一标签
    reorganized_data = []
    
    # 为每个情感标签创建完整的batch
    for sentiment, items in sentiment_groups.items():
        if items:  # 只处理非空的组
            # 将该情感的样本按batch_size分组
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                reorganized_data.extend(batch_items)
                print(f"   添加{sentiment}批次: {len(batch_items)}个样本")
    
    print(f"✅ 数据重组完成: {len(reorganized_data)}个样本")
    print(f"🎯 预期batch数: {len(reorganized_data) // batch_size}")
    
    return reorganized_data

def prepare_grpo_dataset(data_list):
    """准备GRPO训练数据集"""
    print("🔄 准备GRPO数据集...")
    
    # 为GRPO添加prompt字段，并删除messages字段避免冲突
    for item in data_list:
        if 'messages' in item and len(item['messages']) > 0:
            # 提取用户消息作为prompt
            user_message = item['messages'][0]['content']
            item['prompt'] = user_message
            # 删除messages字段避免与GRPO冲突
            del item['messages']
    
    dataset = Dataset.from_list(data_list)
    print(f"✅ GRPO数据集准备完成，共{len(dataset)}个样本")
    
    return dataset

def validate_dataset_format(dataset):
    """验证数据集格式"""
    print("🔍 验证数据格式...")
    sample_item = dataset[0]
    print(f"   数据字段: {list(sample_item.keys())}")
    if 'prompt' in sample_item:
        print(f"   示例prompt长度: {len(sample_item['prompt'])}")
    if 'sentiment' in sample_item:
        print(f"   示例情感标签: {sample_item['sentiment']}")
    print("✅ 数据格式验证完成")
    
    return True

def create_optimized_dataset(file_path, max_samples=None, batch_size=4):
    """创建优化的数据集（完整流程）"""
    # 加载数据
    data = load_synthesis_data(file_path, max_samples)
    print(f"🔄 原始数据加载完成，共{len(data)}个样本")
    
    # 应用情感分组优化
    print("🧠 应用情感分组优化...")
    grouped_data = create_sentiment_grouped_dataset(data, batch_size)
    print(f"✅ 情感分组完成，优化后数据量: {len(grouped_data)}个样本")
    
    # 准备GRPO数据集
    dataset = prepare_grpo_dataset(grouped_data)
    
    # 验证格式
    validate_dataset_format(dataset)
    
    return dataset, grouped_data