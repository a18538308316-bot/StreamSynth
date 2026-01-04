#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
属性处理模块 - AttrPrompt属性加载和合规性计算
"""
import os
import json
import re
import random
from typing import Dict, List, Any

# AttrPrompt属性配置
ATTRPROMPT_CONFIG = {
    'base_path': '/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/TRL-GRPO/yelp/attrprompt/gpt-3.5-turbo',
    'attributes': ['cuisine', 'subtopics', 'style', 'price_range', 'service_quality', 'atmosphere', 'length'],
    'sentiment_labels': ['very negative', 'negative', 'neutral', 'positive', 'very positive']
}

class AttrPromptAttributeLoader:
    """AttrPrompt属性数据加载器"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.attributes_data = {}
        self.load_all_attributes()
    
    def load_attributes(self, attr_name, classes=None):
        """加载指定属性的数据"""
        attr_path = os.path.join(self.base_path, attr_name)
        
        # 对于通用属性（如cuisine, length, style），加载单个文件
        general_attrs = ['cuisine', 'length', 'style']
        if attr_name in general_attrs:
            # 尝试不同的文件格式
            for ext in ['.txt', '.json']:
                attr_file = os.path.join(attr_path, f"{attr_name}{ext}")
                if os.path.exists(attr_file):
                    try:
                        if ext == '.json':
                            with open(attr_file, 'r', encoding='utf-8') as f:
                                return json.load(f)
                        else:  # .txt format
                            with open(attr_file, 'r', encoding='utf-8') as f:
                                lines = [line.strip() for line in f.readlines() if line.strip()]
                                return lines
                    except Exception as e:
                        print(f"⚠️ 读取属性文件失败: {attr_file}, 错误: {e}")
                        continue
            print(f"⚠️ 属性文件不存在: {attr_path}/{attr_name}.[txt|json]")
            return []
        
        # 对于情感相关属性，需要加载各个情感标签的文件
        else:
            sentiment_data = {}
            for sentiment in ATTRPROMPT_CONFIG['sentiment_labels']:
                # 尝试不同的文件格式
                for ext in ['.txt', '.json']:
                    sentiment_file = os.path.join(attr_path, f"{sentiment.replace(' ', '_')}{ext}")
                    if os.path.exists(sentiment_file):
                        try:
                            if ext == '.json':
                                with open(sentiment_file, 'r', encoding='utf-8') as f:
                                    sentiment_data[sentiment] = json.load(f)
                            else:  # .txt format
                                with open(sentiment_file, 'r', encoding='utf-8') as f:
                                    lines = [line.strip() for line in f.readlines() if line.strip()]
                                    sentiment_data[sentiment] = lines
                            break  # 找到文件就退出格式循环
                        except Exception as e:
                            print(f"⚠️ 读取情感属性文件失败: {sentiment_file}, 错误: {e}")
                            continue
                else:
                    # 如果没有找到文件，使用默认值
                    sentiment_data[sentiment] = []
            return sentiment_data
    
    def load_all_attributes(self):
        """加载所有属性数据"""
        print("🔄 加载AttrPrompt属性数据...")
        
        for attr in ATTRPROMPT_CONFIG['attributes']:
            try:
                self.attributes_data[attr] = self.load_attributes(attr)
                print(f"   ✅ {attr}: 加载完成")
            except Exception as e:
                print(f"   ❌ {attr}: 加载失败 - {e}")
                self.attributes_data[attr] = []
        
        print("✅ AttrPrompt属性数据加载完成")
    
    def get_attribute_keywords(self, attr_name, sentiment=None):
        """获取属性的关键词列表"""
        if attr_name not in self.attributes_data:
            return []
        
        attr_data = self.attributes_data[attr_name]
        
        if isinstance(attr_data, dict) and sentiment:
            return attr_data.get(sentiment, [])
        elif isinstance(attr_data, list):
            return attr_data
        else:
            return []

class AttributeComplianceCalculator:
    """属性要求符合度计算器"""
    
    def __init__(self, attr_loader):
        self.attr_loader = attr_loader
    
    def separate_prompt_and_generation(self, completion, prompt=None):
        """改进的prompt和generation分离逻辑 - 专注于JSON结构提取"""
        try:
            # 方法1: 尝试从input/output JSON结构中提取内容
            json_obj = self._find_input_output_json(completion)
            if json_obj and 'input' in json_obj:
                input_text = json_obj['input']
                if isinstance(input_text, str):
                    # 移除"Text: "前缀（如果存在）
                    if input_text.startswith("Text: "):
                        return input_text[6:].strip()
                    return input_text.strip()
            
            # 方法2: 尝试寻找"Text: "模式
            text_patterns = [
                r'"input"\s*:\s*"Text:\s*([^"]+)"',
                r'"input"\s*:\s*"([^"]+)"',
                r'Text:\s*([^\n}]+)',
            ]
            
            import re
            for pattern in text_patterns:
                match = re.search(pattern, completion, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) > 20:  # 确保提取的内容有意义
                        return extracted
            
            # 方法3: 尝试提取JSON后的第一段有意义的文本
            json_start = completion.find('{')
            if json_start != -1:
                # 找到JSON结束位置
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(completion)):
                    if completion[i] == '{':
                        brace_count += 1
                    elif completion[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end != -1:
                    # 提取JSON后的内容作为fallback
                    remaining = completion[json_end:].strip()
                    if remaining and len(remaining) > 20:
                        # 清理并返回第一段有意义的文本
                        lines = remaining.split('\n')
                        for line in lines:
                            line = line.strip()
                            if len(line) > 20 and not line.startswith('#'):
                                return line
            
            # 方法4: 简单的文本清理作为最后手段
            if len(completion) > 50:
                # 移除明显的代码标记
                cleaned = completion
                patterns_to_remove = [
                    r'```[^`]*```',  # 代码块
                    r'def\s+\w+\([^)]*\):',  # 函数定义
                    r'import\s+\w+',  # import语句
                    r'#.*$',  # 注释
                ]
                
                for pattern in patterns_to_remove:
                    cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
                
                # 取前200个字符作为生成内容
                cleaned = cleaned.strip()
                if len(cleaned) > 20:
                    return cleaned[:200] + "..." if len(cleaned) > 200 else cleaned
            
            # 如果所有方法都失败，返回截断的原始内容
            return completion[:100] + "..." if len(completion) > 100 else completion
            
        except Exception as e:
            print(f"⚠️ 文本分离失败: {e}")
            return completion[:100] + "..." if len(completion) > 100 else completion

    def extract_text_from_completion(self, completion, prompt=None):
        """从completion中提取实际的评论文本"""
        try:
            # 导入全局函数
            from .novelsum_diversity import separate_prompt_and_generation_global
            
            # 首先分离出真正的生成内容
            if prompt is not None:
                generation = self.separate_prompt_and_generation(completion, prompt)
            else:
                generation = separate_prompt_and_generation_global(completion)
            
            # 尝试从JSON中提取
            if "{" in generation and "}" in generation:
                start_idx = generation.find("{")
                end_idx = generation.rfind("}") + 1
                json_str = generation[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                input_text = parsed_data.get("input", "")
                if input_text.startswith("Text: "):
                    return input_text[6:]
                return input_text
        except:
            pass
        
        # 如果JSON解析失败，返回处理后的文本
        return (generation if 'generation' in locals() else completion)

    def extract_sentiment_from_json(self, completion):
        """从JSON格式的completion中提取情感标签 - 基于input/output字段的智能提取"""
        try:
            # 方法1: 寻找包含"input"和"output"字段的JSON结构
            json_obj = self._find_input_output_json(completion)
            if json_obj:
                return self._extract_sentiment_from_parsed_json(json_obj)
            
            # 方法2: 如果方法1失败，尝试传统的JSON提取
            json_obj = self._find_first_complete_json(completion)
            if json_obj:
                return self._extract_sentiment_from_parsed_json(json_obj)
            
            # 方法3: 备用提取
            return self._fallback_sentiment_extraction(completion)
                
        except Exception as e:
            print(f"⚠️ JSON解析失败: {str(e)[:50]}")
            return self._fallback_sentiment_extraction(completion)
    
    def _find_input_output_json(self, text):
        """寻找包含input和output字段的JSON对象"""
        import re
        import json
        
        # 寻找"input"关键字的位置
        input_matches = list(re.finditer(r'"input"\s*:', text, re.IGNORECASE))
        
        for input_match in input_matches:
            input_pos = input_match.start()
            
            # 向前寻找最近的开括号{
            json_start = -1
            for i in range(input_pos - 1, -1, -1):
                if text[i] == '{':
                    json_start = i
                    break
                elif text[i] == '}':  # 如果遇到}，说明这个input不属于我们要找的JSON
                    break
            
            if json_start == -1:
                continue
            
            # 从json_start开始寻找完整的JSON
            json_obj = self._extract_json_from_position(text, json_start)
            if json_obj and 'input' in json_obj and 'output' in json_obj:
                return json_obj
        
        return None
    
    def _find_first_complete_json(self, text):
        """寻找第一个完整的JSON对象"""
        import json
        
        json_start = text.find('{')
        if json_start == -1:
            return None
        
        return self._extract_json_from_position(text, json_start)
    
    def _extract_json_from_position(self, text, start_pos):
        """从指定位置开始提取完整的JSON对象"""
        import json
        
        try:
            # 寻找匹配的结束大括号
            brace_count = 0
            json_end = -1
            
            for i in range(start_pos, len(text)):
                char = text[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end == -1:
                return None
            
            # 提取并清理JSON字符串
            json_str = text[start_pos:json_end]
            json_str = self._clean_json_string_simple(json_str)
            
            # 尝试解析
            return json.loads(json_str)
            
        except:
            return None
    
    def _clean_json_string_simple(self, json_str):
        """简单的JSON字符串清理"""
        # 移除控制字符但保留基本格式
        import re
        
        # 移除危险的控制字符
        json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
        
        # 规范化空白字符
        json_str = re.sub(r'\s+', ' ', json_str)
        
        return json_str.strip()
    
    def _extract_sentiment_from_parsed_json(self, json_obj):
        """从已解析的JSON对象中提取情感标签"""
        if not isinstance(json_obj, dict):
            return ""
        
        # 优先查找output字段
        output_keys = ['output', 'sentiment', 'response', 'result', 'answer']
        for key in output_keys:
            if key in json_obj:
                output = json_obj[key]
                if isinstance(output, list):
                    output = output[0] if output else ""
                elif output is not None:
                    output = str(output).strip()
                
                # 验证是否是有效的情感标签
                if self._is_valid_sentiment(output):
                    return output
        
        # 如果没有找到标准键，搜索包含情感词的值
        for key, value in json_obj.items():
            if isinstance(value, str):
                value = value.strip()
                if self._is_valid_sentiment(value):
                    return value
        
        return ""
    
    def _is_valid_sentiment(self, text):
        """检查文本是否是有效的情感标签"""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        valid_sentiments = [
            'very negative', 'very positive', 'negative', 'positive', 'neutral'
        ]
        
        return text_lower in valid_sentiments
    
    def _fallback_sentiment_extraction(self, completion):
        """备用情感提取方法，使用简单的字符串匹配"""
        # 直接搜索情感词，不使用复杂的正则表达式
        sentiments = ['very negative', 'very positive', 'negative', 'positive', 'neutral']
        
        completion_lower = completion.lower()
        for sentiment in sentiments:
            if sentiment in completion_lower:
                return sentiment
        
        return ""

    def calculate_sentiment_consistency(self, completion, target_sentiment):
        """计算情感标签一致性"""
        extracted_sentiment = self.extract_sentiment_from_json(completion)
        
        # 修复：确保extracted_sentiment是字符串
        if isinstance(extracted_sentiment, list):
            extracted_sentiment = extracted_sentiment[0] if extracted_sentiment else ""
        elif not isinstance(extracted_sentiment, str):
            extracted_sentiment = str(extracted_sentiment) if extracted_sentiment else ""
        
        if not extracted_sentiment:
            return 0.0
        
        # 精确匹配
        if extracted_sentiment.lower() == target_sentiment.lower():
            return 1.0
        
        # 部分匹配（例如：positive vs very positive）
        similarity_mapping = {
            'very negative': ['negative', 'very_negative'],
            'negative': ['very negative', 'very_negative'],
            'neutral': ['mixed', 'average', 'okay'],
            'positive': ['very positive', 'very_positive', 'good'],
            'very positive': ['positive', 'very_positive', 'excellent']
        }
        
        target_lower = target_sentiment.lower()
        extracted_lower = extracted_sentiment.lower()
        
        if target_lower in similarity_mapping:
            if any(similar in extracted_lower for similar in similarity_mapping[target_lower]):
                return 0.7
        
        return 0.0  # 完全不匹配
    
    def calculate_cuisine_compliance(self, completion, target_cuisine):
        """计算菜系匹配度"""
        if not target_cuisine or target_cuisine.lower() == 'default_value':
            return 0.5
        
        completion_lower = completion.lower()
        target_lower = target_cuisine.lower()
        
        # 直接匹配菜系名称
        if target_lower in completion_lower:
            return 1.0
        
        # 菜系相关关键词匹配
        cuisine_keywords = {
            # 已有的菜系
            'mexican': ['taco', 'burrito', 'salsa', 'guacamole', 'quesadilla', 'enchilada', 'fajita'],
            'chinese': ['dim sum', 'noodles', 'rice', 'stir fry', 'dumpling', 'wontons', 'fried rice'],
            'italian': ['pasta', 'pizza', 'marinara', 'parmesan', 'risotto', 'lasagna', 'ravioli'],
            'indian': ['curry', 'naan', 'biryani', 'tandoor', 'masala', 'dal', 'samosa'],
            'thai': ['pad thai', 'tom yum', 'coconut', 'lemongrass', 'basil', 'curry', 'satay'],
            'japanese': ['sushi', 'sashimi', 'ramen', 'tempura', 'miso', 'udon', 'teriyaki'],
            'american': ['burger', 'fries', 'bbq', 'sandwich', 'steak', 'ribs', 'wings'],
            'french': ['croissant', 'baguette', 'wine', 'cheese', 'bistro', 'ratatouille', 'crêpe'],
            'portuguese': ['bacalhau', 'cod', 'pastéis', 'francesinha', 'caldo verde', 'sardines', 'port wine', 'custard tart'],
            'brazilian': ['churrasco', 'picanha', 'feijoada', 'pão de açúcar', 'caipirinha', 'açaí'],
            'spanish': ['paella', 'tapas', 'gazpacho', 'sangria', 'chorizo', 'jamón'],
            'korean': ['kimchi', 'bulgogi', 'bibimbap', 'korean bbq', 'galbi', 'banchan'],
            'vietnamese': ['pho', 'banh mi', 'spring rolls', 'vermicelli', 'fish sauce'],
            'greek': ['gyros', 'tzatziki', 'moussaka', 'feta', 'olive oil', 'souvlaki'],
            
            # 新增菜系
            'turkish': ['kebab', 'döner', 'baklava', 'turkish delight', 'börek', 'lahmacun'],
            'cajun': ['jambalaya', 'gumbo', 'crawfish', 'beignet', 'andouille', 'etouffee'],
            'tex-mex': ['nachos', 'fajitas', 'chili con carne', 'queso', 'jalapeño', 'chimichangas'],
            'peruvian': ['ceviche', 'quinoa', 'anticuchos', 'pisco', 'aji amarillo', 'causa'],
            'argentinean': ['empanadas', 'asado', 'chimichurri', 'malbec', 'dulce de leche'],
            'colombian': ['arepa', 'bandeja paisa', 'sancocho', 'aguardiente', 'patacon'],
            'venezuelan': ['cachapa', 'hallaca', 'tequeños', 'pabellón', 'chicha'],
            'ethiopian': ['injera', 'berbere', 'doro wat', 'kitfo', 'tej', 'coffee ceremony'],
            'moroccan': ['tagine', 'couscous', 'harissa', 'pastilla', 'mint tea', 'preserved lemon'],
            'south african': ['biltong', 'boerewors', 'bobotie', 'sosaties', 'potjiekos'],
            'nigerian': ['jollof rice', 'suya', 'egusi', 'plantain', 'pepper soup', 'fufu'],
            'egyptian': ['ful medames', 'koshari', 'molokheya', 'falafel', 'tahini'],
            'malaysian': ['laksa', 'rendang', 'char kway teow', 'satay', 'roti canai'],
            'singaporean': ['hainanese chicken rice', 'laksa', 'char siu', 'chili crab'],
            'indonesian': ['nasi goreng', 'satay', 'rendang', 'gado-gado', 'sambal'],
            'lebanese': ['hummus', 'tabbouleh', 'fattoush', 'kibbeh', 'shawarma', 'baklava'],
            'mediterranean': ['olive oil', 'hummus', 'pita', 'feta', 'olives', 'grilled fish'],
            'middle eastern': ['hummus', 'falafel', 'shawarma', 'pita', 'tahini', 'za\'atar'],
            'russian': ['borscht', 'pierogi', 'caviar', 'vodka', 'beef stroganoff', 'blini'],
            'german': ['schnitzel', 'sauerkraut', 'bratwurst', 'pretzel', 'beer', 'strudel'],
            'polish': ['pierogi', 'kielbasa', 'golumpki', 'bigos', 'kapusta'],
            'scandinavian': ['lutefisk', 'gravlax', 'aquavit', 'meatballs', 'lingonberry'],
            'austrian': ['schnitzel', 'sachertorte', 'apfelstrudel', 'goulash'],
            'swiss': ['fondue', 'raclette', 'rösti', 'cheese', 'chocolate'],
            'canadian': ['poutine', 'maple syrup', 'tourtière', 'butter tart', 'nanaimo bar'],
            'australian': ['meat pie', 'vegemite', 'lamington', 'pavlova', 'barbie'],
            'new zealand': ['pavlova', 'hangi', 'green-lipped mussel', 'hokey pokey'],
            'polynesian': ['poi', 'kalua pig', 'lomi lomi', 'taro', 'coconut'],
            'hawaiian': ['poke', 'luau', 'kalua pig', 'lomi lomi salmon', 'shave ice', 'spam musubi']
        }
        
        if target_lower in cuisine_keywords:
            keywords = cuisine_keywords[target_lower]
            matches = sum(1 for keyword in keywords if keyword in completion_lower)
            return min(matches * 0.3, 1.0)
        
        return 0.0
    
    def calculate_length_compliance(self, completion, target_length, tolerance=25):
        """计算长度符合度 - 支持范围和单一目标值"""
        try:
            # 从completion中提取实际文本
            text_content = self.extract_text_from_completion(completion)
            actual_length = len(text_content.split())
            
            # 调试信息
            print(f"DEBUG: target_length类型: {type(target_length)}, 值: {target_length}")
            print(f"DEBUG: actual_length: {actual_length}")
            
            # 处理不同格式的target_length
            if isinstance(target_length, dict):
                # 新格式：包含范围信息
                min_length = target_length['min']
                max_length = target_length['max']
                target_value = target_length['target']
                
                # 如果在原始要求范围内，给满分
                if min_length <= actual_length <= max_length:
                    return 1.0
                
                # 如果在扩展容忍范围内，根据距离给分
                extended_min = min_length - tolerance
                extended_max = max_length + tolerance
                
                if extended_min <= actual_length <= extended_max:
                    if actual_length < min_length:
                        # 短于要求范围
                        diff = min_length - actual_length
                        score = 1.0 - (diff / tolerance) * 0.5
                    else:
                        # 长于要求范围
                        diff = actual_length - max_length
                        score = 1.0 - (diff / tolerance) * 0.5
                    return max(score, 0.5)
                else:
                    return 0.0
                    
            else:
                # 旧格式：单一目标值 - 这里可能是问题所在
                print(f"DEBUG: 使用旧格式，target_length: {target_length}")
                target_value = int(target_length) if isinstance(target_length, str) else target_length
                length_diff = abs(actual_length - target_value)
                
                if length_diff <= tolerance:
                    # 在容忍范围内，根据差异给分
                    score = 1.0 - (length_diff / tolerance) * 0.3
                    return max(score, 0.7)
                elif length_diff <= tolerance * 2:
                    # 超出容忍范围但不太多
                    return 0.5
                else:
                    return 0.0  # 超出范围太多
        except Exception as e:
            print(f"DEBUG: calculate_length_compliance异常: {e}")
            return 0.0
    
    def calculate_attribute_keyword_match(self, completion, attr_name, target_value, sentiment=None):
        """计算属性关键词匹配度"""
        if not target_value or target_value.lower() == 'default_value':
            return 0.5
        
        completion_lower = completion.lower()
        target_lower = target_value.lower()
        
        # 直接匹配
        if target_lower in completion_lower:
            return 1.0
        
        # 特定属性的关键词匹配
        score = self._calculate_specific_attribute_match(completion_lower, attr_name, target_lower)
        if score > 0:
            return score
        
        # 获取属性相关关键词（从数据文件）
        keywords = self.attr_loader.get_attribute_keywords(attr_name, sentiment)
        
        if not keywords:
            return 0.0
        
        # 关键词匹配度计算
        matches = 0
        for keyword in keywords:
            if isinstance(keyword, str) and keyword.lower() in completion_lower:
                matches += 1
        
        return min(matches * 0.2, 1.0)
    
    def _calculate_specific_attribute_match(self, completion_lower, attr_name, target_lower):
        """针对特定属性的关键词匹配"""
        
        # Style属性匹配
        if attr_name == 'style':
            style_keywords = {
                'descriptive': ['detailed', 'vivid', 'description', 'describes', 'picture'],
                'personal narrative': ['i went', 'my experience', 'we visited', 'personal', 'story'],
                'analysis': ['quality', 'evaluate', 'assess', 'rating', 'pros and cons'],
                'comparative': ['compared to', 'similar to', 'better than', 'versus', 'like other'],
                'gastronomic': ['flavor', 'taste', 'seasoning', 'cooking', 'preparation'],
                'casual conversational': ['honestly', 'so', 'really', 'pretty good', 'not bad'],
                'critical professional': ['professional', 'critique', 'evaluation', 'standards'],
                'enthusiastic emotional': ['amazing', 'fantastic', 'terrible', 'incredible', 'love'],
                'storytelling': ['arrived', 'started', 'then', 'finally', 'journey', 'experience'],
                'humorous': ['funny', 'joke', 'hilarious', 'amusing', 'witty']
            }
            
            for style_type, keywords in style_keywords.items():
                if style_type in target_lower:
                    matches = sum(1 for kw in keywords if kw in completion_lower)
                    if matches > 0:
                        return min(matches * 0.3, 1.0)
        
        # Price range属性匹配
        elif attr_name == 'price_range':
            price_keywords = {
                'budget-friendly': ['cheap', 'affordable', 'budget', 'inexpensive', '$'],
                'very affordable': ['very cheap', 'super affordable', 'bargain', 'dirt cheap'],
                'reasonably priced': ['reasonable', 'fair price', 'decent price', 'good value'],
                'moderate': ['moderate', 'average price', 'mid-range', 'not too expensive'],
                'slightly expensive': ['bit pricey', 'somewhat expensive', 'higher end'],
                'upscale': ['upscale', 'premium', 'high-end', 'fancy', '$$$'],
                'fine dining': ['fine dining', 'luxury', 'expensive', '$$$$'],
                'premium': ['premium', 'top tier', 'high-end', 'luxury'],
                'luxury': ['luxury', 'extravagant', 'splurge', 'high-end']
            }
            
            for price_type, keywords in price_keywords.items():
                if price_type in target_lower:
                    matches = sum(1 for kw in keywords if kw in completion_lower)
                    if matches > 0:
                        return min(matches * 0.4, 1.0)
        
        # Service quality属性匹配
        elif attr_name == 'service_quality':
            service_keywords = {
                'excellent': ['excellent', 'outstanding', 'exceptional', 'perfect', 'amazing'],
                'good': ['good', 'great', 'friendly', 'attentive', 'helpful'],
                'hit-or-miss': ['inconsistent', 'mixed', 'sometimes good', 'hit or miss'],
                'poor': ['poor', 'bad', 'terrible', 'rude', 'slow', 'inattentive'],
                'fast': ['quick', 'fast', 'prompt', 'speedy'],
                'slow': ['slow', 'took forever', 'waited long', 'delayed']
            }
            
            for service_type, keywords in service_keywords.items():
                if service_type in target_lower:
                    matches = sum(1 for kw in keywords if kw in completion_lower)
                    if matches > 0:
                        return min(matches * 0.4, 1.0)
        
        # Atmosphere属性匹配
        elif attr_name == 'atmosphere':
            atmosphere_keywords = {
                'cozy': ['cozy', 'warm', 'intimate', 'comfortable'],
                'elegant': ['elegant', 'sophisticated', 'classy', 'upscale'],
                'casual': ['casual', 'relaxed', 'laid-back', 'informal'],
                'romantic': ['romantic', 'intimate', 'candlelit', 'date night'],
                'family-friendly': ['family', 'kids', 'children', 'families'],
                'lively': ['lively', 'energetic', 'bustling', 'vibrant'],
                'quiet': ['quiet', 'peaceful', 'calm', 'serene'],
                'nothing special': ['nothing special', 'ordinary', 'basic', 'unremarkable', 'plain']
            }
            
            for atm_type, keywords in atmosphere_keywords.items():
                if atm_type in target_lower:
                    matches = sum(1 for kw in keywords if kw in completion_lower)
                    if matches > 0:
                        return min(matches * 0.4, 1.0)
        
        return 0.0

def extract_attributes_from_input(input_text):
    """从测试数据的input字段中提取所有属性信息"""
    attributes = {}
    
    # 提取目标情感标签 - 修复正则表达式匹配实际格式
    target_patterns = [
        r"The overall review should be ([a-zA-Z\s]+)",
        r"Generate exactly one review with ([a-zA-Z\s]+) sentiment",
        r"sentiment label exactly as provided[^:]*:\s*([a-zA-Z\s]+)",
        r"Target sentiment for generation:\s*([a-zA-Z\s]+)"
    ]
    
    for pattern in target_patterns:
        target_match = re.search(pattern, input_text, re.IGNORECASE)
        if target_match:
            sentiment = target_match.group(1).strip()
            # 标准化情感标签
            sentiment_mapping = {
                'very negative': 'very negative',
                'negative': 'negative', 
                'neutral': 'neutral',
                'positive': 'positive',
                'very positive': 'very positive'
            }
            attributes['target_sentiment'] = sentiment_mapping.get(sentiment.lower(), sentiment)
            break
    
    # 如果还没找到，尝试更宽泛的搜索
    if 'target_sentiment' not in attributes:
        # 搜索包含情感词的句子
        sentiment_words = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        for word in sentiment_words:
            if word in input_text.lower():
                attributes['target_sentiment'] = word
                break
    
    # Amazon数据集不需要提取餐厅类型和子主题
    # 这些是Yelp数据集特有的属性
    
    # 提取长度要求
    length_pattern = r"Should be in length between (\d+) words and (\d+) words"
    length_match = re.search(length_pattern, input_text)
    if length_match:
        min_length, max_length = int(length_match.group(1)), int(length_match.group(2))
        # 存储完整的长度范围信息
        attributes['length'] = {
            'min': min_length,
            'max': max_length,
            'target': (min_length + max_length) // 2
        }
    
    # 提取写作风格
    style_pattern = r"The style of the review should be '([^']+)'"
    style_match = re.search(style_pattern, input_text)
    if style_match:
        attributes['style'] = style_match.group(1).strip()
    
    # 提取价格方面
    price_pattern = r"The pricing aspect should reflect '([^']+)'"
    price_match = re.search(price_pattern, input_text)
    if price_match:
        attributes['price_range'] = price_match.group(1).strip()
    
    # 提取服务质量
    service_pattern = r"The service quality should be described as '([^']+)'"
    service_match = re.search(service_pattern, input_text)
    if service_match:
        attributes['service_quality'] = service_match.group(1).strip()
    
    # 提取氛围描述
    atmosphere_pattern = r"The atmosphere should be portrayed as '([^']+)'"
    atmosphere_match = re.search(atmosphere_pattern, input_text)
    if atmosphere_match:
        attributes['atmosphere'] = atmosphere_match.group(1).strip()
    
    return attributes

def generate_random_prompt_attributes(attr_loader):
    """生成随机的提示属性组合（兼容性函数，仅在无法从input提取时使用）"""
    if not attr_loader:
        return {}
    
    # 随机选择一个情感标签
    target_sentiment = random.choice(ATTRPROMPT_CONFIG['sentiment_labels'])
    
    # 生成属性组合
    attributes = {}
    
    for attr_name in ATTRPROMPT_CONFIG['attributes']:
        if attr_name == 'length':
            attributes[attr_name] = random.choice([100, 150, 200, 250, 300])
        else:
            keywords = attr_loader.get_attribute_keywords(attr_name, target_sentiment)
            if keywords:
                if isinstance(keywords, list) and keywords:
                    attributes[attr_name] = random.choice(keywords)
                elif isinstance(keywords, dict):
                    # 对于嵌套字典，随机选择一个值
                    all_values = []
                    for v in keywords.values():
                        if isinstance(v, list):
                            all_values.extend(v)
                    if all_values:
                        attributes[attr_name] = random.choice(all_values)
                else:
                    attributes[attr_name] = 'default_value'
            else:
                attributes[attr_name] = 'default_value'
    
    attributes['target_sentiment'] = target_sentiment
    return attributes