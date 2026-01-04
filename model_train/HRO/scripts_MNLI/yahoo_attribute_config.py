#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yahoo dataset attribute configuration.
Based on the attribute definitions in ``yahoo_data_synthesis_attrprompt_type.py``.
"""

# Base directory for Yahoo attribute files (update this to your local path)
YAHOO_ATTRIBUTE_BASE_PATH = "/path/to/yahoo_property"

# Yahoo数据集标签
YAHOO_LABELS = [
    "Society & Culture",
    "Science & Mathematics", 
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government"
]

class YahooAttrPromptConfig:
    """Attribute prompt configuration for the Yahoo dataset (MNLI usage)."""
    
    def __init__(self):
        self.base_path = YAHOO_ATTRIBUTE_BASE_PATH
        
        # Define all attributes and their corresponding file paths
        # Yahoo dataset uses txt files to store attribute values
        self.attributes = {
            'question_type': f"{self.base_path}/question_type/question_type.txt",
            'user_intent': f"{self.base_path}/user_intent/user_intent.txt",
            'answer_tone': f"{self.base_path}/answer_tone/answer_tone.txt",
            'complexity_level': f"{self.base_path}/complexity_level/complexity_level.txt",
            'evidence_expectation': f"{self.base_path}/evidence_expectation/evidence_expectation.txt",
            'style': f"{self.base_path}/style/style.txt",
            'length': f"{self.base_path}/length/length.txt"
        }
        
        # Label-specific attributes stored as jsonl files (e.g. domain_subtopic)
        self.label_specific_attributes = {
            'domain_subtopic': f"{self.base_path}/domain_subtopic"
        }
        
        # Yahoo dataset labels
        self.labels = YAHOO_LABELS
        
        # Attribute matching keywords (simple keyword detection)
        self.attribute_keywords = {}
        
        # Load txt-based attributes
        for attr_name in self.attributes:
            self.attribute_keywords[attr_name] = self._load_attribute_values(attr_name)
        
        # Initialize cache for label-specific attributes (loaded lazily)
        for attr_name in self.label_specific_attributes:
            self.attribute_keywords[attr_name] = {}
    
    def _load_attribute_values(self, attribute_name):
        """Load possible values for a given attribute from a txt file."""
        try:
            file_path = self.attributes[attribute_name]
            with open(file_path, 'r', encoding='utf-8') as f:
                values = [line.strip() for line in f if line.strip()]
            return values
        except Exception as e:
            print(f"Warning: Failed to load {attribute_name} values: {e}")
            return []
    
    def _load_label_specific_values(self, attribute_name, label):
        """Load label-specific attribute values from jsonl files."""
        if attribute_name not in self.label_specific_attributes:
            return []
        
        # If values for this label were already loaded, return cached ones
        if label in self.attribute_keywords[attribute_name]:
            return self.attribute_keywords[attribute_name][label]
        
        try:
            # Convert human-readable label into a file name
            label_file_map = {
                "Society & Culture": "society_culture.jsonl",
                "Science & Mathematics": "science_mathematics.jsonl",
                "Health": "health.jsonl",
                "Education & Reference": "education_reference.jsonl",
                "Computers & Internet": "computers_internet.jsonl",
                "Sports": "sports.jsonl",
                "Business & Finance": "business_finance.jsonl",
                "Entertainment & Music": "entertainment_music.jsonl",
                "Family & Relationships": "family_relationships.jsonl",
                "Politics & Government": "politics_government.jsonl"
            }
            
            file_name = label_file_map.get(label, 'society_culture.jsonl')
            file_path = f"{self.label_specific_attributes[attribute_name]}/{file_name}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                values = [line.strip() for line in f if line.strip()]
            
            # Cache the result
            self.attribute_keywords[attribute_name][label] = values
            return values
            
        except Exception as e:
            print(f"Warning: Failed to load {attribute_name} values for label {label}: {e}")
            return []
    
    def get_attribute_check_function(self, attribute_name, target_value, label=None):
        """Return a checker/scoring function for a given attribute (optionally label-specific)."""
        
        # Special handling for length attributes
        if attribute_name == 'length':
            return self._create_length_checker(target_value)
        
        # Handle label-specific attributes
        if attribute_name in self.label_specific_attributes:
            if not label:
                return lambda text: 0.3  # Default fallback score
            
            # Get attribute values for this label
            attribute_values = self._load_label_specific_values(attribute_name, label)
            
            def check_label_specific_attribute(text):
                text_lower = text.lower()
                
                # Check whether any attribute value appears in the text
                for value in attribute_values:
                    if value.lower() in text_lower:
                        return 1.0
                
                # Partial match check
                partial_matches = 0
                for value in attribute_values:
                    value_words = value.lower().split()
                    matches = sum(1 for word in value_words if word in text_lower and len(word) > 2)
                    if matches > 0:
                        partial_matches += matches / len(value_words)
                
                return min(partial_matches / len(attribute_values), 1.0) if attribute_values else 0.3
            
            return check_label_specific_attribute
        
        # Handle regular txt-based attributes
        if attribute_name not in self.attributes:
            return lambda text: 0
        
        attribute_values = self.attribute_keywords[attribute_name]
        
        if not attribute_values:
            return lambda text: 0
        
        # Generic attributes: check if target keywords appear in the text
        def check_attribute_in_text(text):
            text_lower = text.lower()
            
            # 精确匹配目标值
            target_lower = target_value.lower()
            if target_lower in text_lower:
                return 1.0
            
            # Check each component of the target value
            target_parts = target_lower.split('_')
            matches = 0
            for part in target_parts:
                if part in text_lower and len(part) > 2:  # Avoid overly short words
                    matches += 1
            
            return matches / len(target_parts) if target_parts else 0
        
        return check_attribute_in_text
    
    def _create_length_checker(self, target_length):
        """Create a checker function based on target text length."""
        try:
            target_num = int(target_length)
            def check_length(text):
                word_count = len(text.split())
                if word_count >= target_num * 0.8 and word_count <= target_num * 1.2:
                    return 1.0
                elif word_count >= target_num * 0.6 and word_count <= target_num * 1.4:
                    return 0.7
                elif word_count >= target_num * 0.4 and word_count <= target_num * 1.6:
                    return 0.4
                else:
                    return 0.1
            return check_length
        except (ValueError, TypeError):
            return lambda text: 0.5

# Create a global configuration instance
YAHOO_ATTRPROMPT_CONFIG = YahooAttrPromptConfig()

def load_yahoo_sample_attributes(sample_input):
    """Load Yahoo attribute constraints from a sample input."""
    from .attribute_handler import extract_attributes_from_input
    return extract_attributes_from_input(sample_input)

def get_yahoo_labels():
    """Return all Yahoo dataset labels."""
    return YAHOO_LABELS

def get_yahoo_attributes():
    """Return all Yahoo dataset attribute names."""
    return list(YAHOO_ATTRPROMPT_CONFIG.attributes.keys()) + list(YAHOO_ATTRPROMPT_CONFIG.label_specific_attributes.keys())
