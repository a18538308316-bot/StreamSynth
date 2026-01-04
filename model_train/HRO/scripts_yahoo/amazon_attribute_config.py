#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon数据集属性配置文件
基于amazon_data_synthesis_attrprompt_type.py中的属性定义
"""

# Amazon数据集属性基础路径
AMAZON_ATTRIBUTE_BASE_PATH = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/TRL-GRPO-ohter-dataset/amazon/amazon_property"


class AmazonAttrPromptConfig:
    """Amazon数据集属性提示词配置"""
    
    def __init__(self):
        self.base_path = AMAZON_ATTRIBUTE_BASE_PATH
        
        # 定义所有属性及其文件路径
        # 注意：部分属性使用txt文件，部分属性使用按情感标签的jsonl文件
        self.attributes = {
            'product_category': f"{self.base_path}/product_category/product_category.txt",
            'usage_context': f"{self.base_path}/usage_context/usage_context.txt", 
            'review_angle': f"{self.base_path}/review_angle/review_angle.txt",
            'feature_focus': f"{self.base_path}/feature_focus/feature_focus.txt",
            'style': f"{self.base_path}/style/style.txt",
            'length': f"{self.base_path}/length/length.txt"
        }
        
        # 按情感标签的jsonl文件属性
        self.sentiment_specific_attributes = {
            'price_perception': f"{self.base_path}/price_perception",
            'product_quality': f"{self.base_path}/product_quality",
            'packaging': f"{self.base_path}/packaging",
            'delivery_experience': f"{self.base_path}/delivery_experience",
            'customer_support': f"{self.base_path}/customer_support",
            'overall_experience': f"{self.base_path}/overall_experience"
        }
        
        # 情感标签（与原来一致）
        self.sentiment_labels = [
            'very negative', 'negative', 'neutral', 'positive', 'very positive'
        ]
        
        # 属性匹配的容错关键词（简单的关键词检测）
        self.attribute_keywords = {}
        
        # 加载txt文件属性
        for attr_name in self.attributes:
            self.attribute_keywords[attr_name] = self._load_attribute_values(attr_name)
        
        # 加载按情感标签的jsonl文件属性（初始化为空，按需加载）
        for attr_name in self.sentiment_specific_attributes:
            self.attribute_keywords[attr_name] = {}
    
    def _load_attribute_values(self, attribute_name):
        """加载指定属性的可选值（txt文件）"""
        try:
            file_path = self.attributes[attribute_name]
            with open(file_path, 'r', encoding='utf-8') as f:
                values = [line.strip() for line in f if line.strip()]
            return values
        except Exception as e:
            print(f"Warning: Failed to load {attribute_name} values: {e}")
            return []
    
    def _load_sentiment_specific_values(self, attribute_name, sentiment):
        """加载按情感标签的jsonl文件属性值"""
        if attribute_name not in self.sentiment_specific_attributes:
            return []
        
        # 如果已经加载过，直接返回
        if sentiment in self.attribute_keywords[attribute_name]:
            return self.attribute_keywords[attribute_name][sentiment]
        
        try:
            # 将情感标签转换为文件名格式
            sentiment_file_map = {
                'very negative': 'very_negative.jsonl',
                'negative': 'negative.jsonl',
                'neutral': 'neutral.jsonl',
                'positive': 'positive.jsonl',
                'very positive': 'very_positive.jsonl'
            }
            
            file_name = sentiment_file_map.get(sentiment, 'neutral.jsonl')
            file_path = f"{self.sentiment_specific_attributes[attribute_name]}/{file_name}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                values = [line.strip() for line in f if line.strip()]
            
            # 缓存结果
            self.attribute_keywords[attribute_name][sentiment] = values
            return values
            
        except Exception as e:
            print(f"Warning: Failed to load {attribute_name} values for sentiment {sentiment}: {e}")
            return []
    
    def get_attribute_check_function(self, attribute_name, target_value, sentiment=None):
        """获取指定属性的检查函数（支持情感特异性属性）"""
        
        # 特殊处理长度属性
        if attribute_name == 'length':
            return self._create_length_checker(target_value)
        
        # 处理情感特异性属性
        if attribute_name in self.sentiment_specific_attributes:
            if not sentiment:
                return lambda text: 0.3  # 默认分数
            
            # 获取该情感下的属性值
            attribute_values = self._load_sentiment_specific_values(attribute_name, sentiment)
            
            def check_sentiment_specific_attribute(text):
                text_lower = text.lower()
                
                # 检查是否包含任何该情感下的属性值
                for value in attribute_values:
                    if value.lower() in text_lower:
                        return 1.0
                
                # 部分匹配检查
                partial_matches = 0
                for value in attribute_values:
                    value_words = value.lower().split()
                    matches = sum(1 for word in value_words if word in text_lower and len(word) > 2)
                    if matches > 0:
                        partial_matches += matches / len(value_words)
                
                return min(partial_matches / len(attribute_values), 1.0) if attribute_values else 0.3
            
            return check_sentiment_specific_attribute
        
        # 处理普通txt文件属性
        if attribute_name not in self.attributes:
            return lambda text: 0
        
        attribute_values = self.attribute_keywords[attribute_name]
        
        if not attribute_values:
            return lambda text: 0
        
        # 一般属性：检查是否包含关键词
        def check_attribute_in_text(text):
            text_lower = text.lower()
            
            # 精确匹配目标值
            target_lower = target_value.lower()
            if target_lower in text_lower:
                return 1.0
            
            # 检查目标值的各个部分
            target_parts = target_lower.split('_')
            matches = 0
            for part in target_parts:
                if part in text_lower and len(part) > 2:  # 避免单词过短
                    matches += 1
            
            return matches / len(target_parts) if target_parts else 0
        
        return check_attribute_in_text
    
    def _create_length_checker(self, target_length):
        """创建长度检查器"""
        try:
            target_num = int(target_length)
        except:
            return lambda text: 0
        
        def check_length(text):
            word_count = len(text.split())
            
            # 根据amazon_data_synthesis_attrprompt_type.py中的长度约束
            length_ranges = {
                40: (30, 50),
                80: (60, 100), 
                120: (100, 140),
                200: (180, 220),
                300: (280, 320),
            }
            
            if target_num in length_ranges:
                min_len, max_len = length_ranges[target_num]
                if min_len <= word_count <= max_len:
                    return 1.0
                elif min_len * 0.8 <= word_count <= max_len * 1.2:  # 10%容错
                    return 0.8
                else:
                    return 0.3  # 长度不对但可能有内容
            
            return 0.2 if word_count > 20 else 0  # 基础长度评分
        
        return check_length


# 全局配置实例
AMAZON_ATTRPROMPT_CONFIG = AmazonAttrPromptConfig()


def get_amazon_attribute_config():
    """获取Amazon属性配置"""
    return AMAZON_ATTRPROMPT_CONFIG


def load_amazon_sample_attributes(sample_input):
    """从样本输入中提取Amazon属性"""
    attributes = {}
    
    # 提取各个属性值（基于提示词格式）
    attr_patterns = {
        'product_category': r'Product category context: ([^,\n]+)',
        'usage_context': r'Primary usage context: ([^,\n]+)',
        'review_angle': r'Review angle perspective: ([^,\n]+)',
        'feature_focus': r'Core feature focus: ([^,\n]+)',
        'style': r'Writing style: ([^,\n]+)',
        'price_perception': r'Perceived price/value: ([^,\n]+)', 
        'product_quality': r'Product quality/build impression: ([^,\n]+)',
        'packaging': r'Packaging condition: ([^,\n]+)',
        'delivery_experience': r'Delivery experience: ([^,\n]+)',
        'customer_support': r'Customer support experience: ([^,\n]+)',
        'overall_experience': r'Overall ownership impression: ([^,\n]+)',
        'length': r'Length between (\d+) and (\d+)'
    }
    
    import re
    
    for attr_name, pattern in attr_patterns.items():
        match = re.search(pattern, sample_input, re.IGNORECASE)
        if match:
            if attr_name == 'length':
                # 长度特殊处理：取两个数字的平均值
                try:
                    min_len, max_len = int(match.group(1)), int(match.group(2))
                    attributes[attr_name] = str((min_len + max_len) // 2)
                except:
                    attributes[attr_name] = '80'  # 默认值
            else:
                attributes[attr_name] = match.group(1).strip()
    
    return attributes
