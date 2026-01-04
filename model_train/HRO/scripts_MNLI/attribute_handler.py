import re
import json

def extract_attributes_from_input(input_text):
    """从输入文本中提取MNLI属性约束"""
    attributes = {}
    
    # 提取目标标签
    target_patterns = [
        r'Target label \(must match exactly\):\s*([^\n]+)',
        r'Target label:\s*([^\n]+)',
        r'label \(must match exactly\):\s*([^\n]+)',
        r'label:\s*([^\n]+)'
    ]
    
    target_label = None
    for pattern in target_patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            target_label = match.group(1).strip()
            break
    
    if target_label:
        # 标准化MNLI标签
        target_label = target_label.lower()
        if target_label in ['entailment', 'contradiction', 'neutral']:
            attributes['target_label'] = target_label
        else:
            attributes['target_label'] = 'neutral'  # 默认值
    else:
        attributes['target_label'] = 'neutral'  # 默认值
    
    # 提取前提域
    domain_patterns = [
        r'Premise domain:\s*([^\n]+)',
        r'premise domain:\s*([^\n]+)'
    ]
    
    for pattern in domain_patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            attributes['premise_domain'] = match.group(1).strip()
            break
    
    # 提取前提风格
    style_patterns = [
        r'Premise style:\s*([^\n]+)',
        r'premise style:\s*([^\n]+)'
    ]
    
    for pattern in style_patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            attributes['premise_style'] = match.group(1).strip()
            break
    
    # 提取假设转换
    transformation_patterns = [
        r'Hypothesis transformation:\s*([^\n]+)',
        r'hypothesis transformation:\s*([^\n]+)'
    ]
    
    for pattern in transformation_patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            attributes['hypothesis_transformation'] = match.group(1).strip()
            break
    
    # 提取语义现象
    phenomenon_patterns = [
        r'Semantic phenomenon:\s*([^\n]+)',
        r'semantic phenomenon:\s*([^\n]+)'
    ]
    
    for pattern in phenomenon_patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            attributes['semantic_phenomenon'] = match.group(1).strip()
            break
    
    # 提取推理类型
    reasoning_patterns = [
        r'Reasoning type:\s*([^\n]+)',
        r'reasoning type:\s*([^\n]+)'
    ]
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            attributes['reasoning_type'] = match.group(1).strip()
            break
    
    # 提取干扰类型
    distraction_patterns = [
        r'Distraction type:\s*([^\n]+)',
        r'distraction type:\s*([^\n]+)'
    ]
    
    for pattern in distraction_patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            attributes['distraction_type'] = match.group(1).strip()
            break
    
    # 提取标签策略
    strategy_patterns = [
        r'Label strategy hint:\s*([^\n]+)',
        r'label strategy hint:\s*([^\n]+)'
    ]
    
    for pattern in strategy_patterns:
        match = re.search(pattern, input_text, re.IGNORECASE)
        if match:
            attributes['label_strategy'] = match.group(1).strip()
            break
    
    # 提取长度约束
    length_patterns = [
        r'Premise length:\s*(\d+)\s*to\s*(\d+)\s*words',
        r'Hypothesis length:\s*(\d+)\s*to\s*(\d+)\s*words'
    ]
    
    for pattern in length_patterns:
        matches = re.findall(pattern, input_text, re.IGNORECASE)
        if matches:
            for match in matches:
                min_len, max_len = int(match[0]), int(match[1])
                if 'Premise length:' in pattern:
                    attributes['length_premise'] = f"{min_len}-{max_len}"
                elif 'Hypothesis length:' in pattern:
                    attributes['length_hypothesis'] = f"{min_len}-{max_len}"
    
    return attributes