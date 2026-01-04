import os
import re
import json

MNLI_ATTRIBUTE_BASE_PATH = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/TRL-GRPO-ohter-dataset/MNLI/MNLI_property"

class MNLIAttrPromptConfig:
    """MNLI数据集属性提示词配置"""

    def __init__(self):
        self.base_path = MNLI_ATTRIBUTE_BASE_PATH

        # MNLI数据集有3个标签
        self.labels = [
            "entailment", "contradiction", "neutral"
        ]

        # 定义所有属性及其文件路径
        self.attributes = {
            'premise_domain': f"{self.base_path}/premise_domain.txt",
            'premise_style': f"{self.base_path}/premise_style.txt",
            'hypothesis_transformation': f"{self.base_path}/hypothesis_transformation.txt",
            'semantic_phenomenon': f"{self.base_path}/semantic_phenomenon.txt",
            'reasoning_type': f"{self.base_path}/reasoning_type.txt",
            'distraction_type': f"{self.base_path}/distraction_type.txt",
            'length_premise': f"{self.base_path}/length_premise.txt",
            'length_hypothesis': f"{self.base_path}/length_hypothesis.txt"
        }

        # 标签特异性属性 (label_strategy)
        self.label_specific_attributes = {
            'label_strategy': f"{self.base_path}/label_strategy"
        }

        self.attribute_keywords = {}

        # 加载txt文件属性
        for attr_name in self.attributes:
            self.attribute_keywords[attr_name] = self._load_attribute_values(attr_name)

        # 加载标签特异性属性（初始化为空，按需加载）
        for attr_name in self.label_specific_attributes:
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

    def _load_label_specific_values(self, attribute_name, label):
        """加载按标签的jsonl文件属性值"""
        if attribute_name not in self.label_specific_attributes:
            return []

        # 如果已经加载过，直接返回
        if label in self.attribute_keywords[attribute_name]:
            return self.attribute_keywords[attribute_name][label]

        try:
            # 将标签转换为文件名格式 (e.g., "entailment" -> "entailment.jsonl")
            file_name = label.lower() + '.jsonl'
            file_path = f"{self.label_specific_attributes[attribute_name]}/{file_name}"

            if not os.path.exists(file_path):
                print(f"Warning: Label-specific attribute file not found: {file_path}")
                return []

            with open(file_path, 'r', encoding='utf-8') as f:
                raw_lines = [line.strip() for line in f if line.strip()]

            parsed_items = []
            for line in raw_lines:
                try:
                    parsed_items.append(json.loads(line))
                except json.JSONDecodeError:
                    # 非JSON格式时退回为纯文本
                    parsed_items.append(line)

            # 提取所有策略值
            all_strategies = []
            for item in parsed_items:
                if isinstance(item, dict) and 'strategies' in item:
                    all_strategies.extend(item['strategies'])
                elif isinstance(item, (list, tuple)):
                    all_strategies.extend(item)
                elif isinstance(item, str):
                    all_strategies.append(item)

            values = [s for s in all_strategies if isinstance(s, str) and s]

            # 缓存结果
            self.attribute_keywords[attribute_name][label] = values
            return values

        except Exception as e:
            print(f"Warning: Failed to load {attribute_name} values for label {label}: {e}")
            return []

    def get_attribute_check_function(self, attribute_name, target_value, label=None):
        """获取指定属性的检查函数（支持标签特异性属性）"""

        # 特殊处理长度属性
        if attribute_name in ['length_premise', 'length_hypothesis']:
            return self._create_length_checker(target_value)

        # 处理标签特异性属性 (label_strategy)
        if attribute_name in self.label_specific_attributes:
            if not label:
                return lambda text: 0.3  # 默认分数

            # 获取该标签下的属性值
            attribute_values = self._load_label_specific_values(attribute_name, label)

            def check_label_specific_attribute(text):
                text_lower = text.lower()
                if not attribute_values:
                    return 0.3 # 没有可检查的属性值，给中等分

                # 检查是否包含任何该标签下的属性值
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

            return check_label_specific_attribute

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
            # target_length 格式为 "min-max" 或 "target_value"
            if isinstance(target_length, dict) and 'target' in target_length:
                target_num = target_length['target']
                min_len = target_length.get('min', target_num * 0.8)
                max_len = target_length.get('max', target_num * 1.2)
            elif isinstance(target_length, str) and '-' in target_length:
                min_len_str, max_len_str = target_length.split('-')
                min_len, max_len = int(min_len_str), int(max_len_str)
                target_num = (min_len + max_len) // 2
            else:
                target_num = int(target_length)
                min_len = target_num * 0.8
                max_len = target_num * 1.2

            def check_length(text):
                word_count = len(text.split())
                if min_len <= word_count <= max_len:
                    return 1.0
                elif (min_len * 0.8 <= word_count <= max_len * 1.2): # 20% 浮动
                    return 0.7
                elif (min_len * 0.6 <= word_count <= max_len * 1.4): # 40% 浮动
                    return 0.4
                else:
                    return 0.1
            return check_length
        except ValueError:
            print(f"Warning: Invalid target_length format: {target_length}. Using default length checker.")
            return lambda text: 0.5 # 无法解析长度，给中等分数

# 创建全局配置实例
MNLI_ATTRPROMPT_CONFIG = MNLIAttrPromptConfig()

def load_mnli_sample_attributes(sample_input):
    """从样本输入中加载MNLI属性约束"""
    from .attribute_handler import extract_attributes_from_input
    return extract_attributes_from_input(sample_input)

def get_mnli_labels():
    """获取MNLI数据集的所有标签"""
    return MNLI_ATTRPROMPT_CONFIG.labels

def get_mnli_attributes():
    """获取MNLI数据集的所有属性"""
    return list(MNLI_ATTRPROMPT_CONFIG.attributes.keys()) + list(MNLI_ATTRPROMPT_CONFIG.label_specific_attributes.keys())
