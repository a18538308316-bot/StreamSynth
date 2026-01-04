#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attribute handling module - AttrPrompt attribute loading and compliance calculation.
"""
import os
import json
import re
import random
from typing import Dict, List, Any

# AttrPrompt configuration
ATTRPROMPT_CONFIG = {
    # Base path for AttrPrompt files (update this to your local path)
    'base_path': '/path/to/yelp_attrprompt/gpt-3.5-turbo',
    'attributes': ['cuisine', 'subtopics', 'style', 'price_range', 'service_quality', 'atmosphere', 'length'],
    'sentiment_labels': ['very negative', 'negative', 'neutral', 'positive', 'very positive']
}

class AttrPromptAttributeLoader:
    """AttrPrompt attribute data loader."""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.attributes_data = {}
        self.load_all_attributes()
    
    def load_attributes(self, attr_name, classes=None):
        """Load data for the specified attribute."""
        attr_path = os.path.join(self.base_path, attr_name)
        
        # For general attributes (such as cuisine, length, style), load from a single file
        general_attrs = ['cuisine', 'length', 'style']
        if attr_name in general_attrs:
            # Try different file formats
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
                        print(f"⚠️ Failed to read attribute file: {attr_file}, error: {e}")
                        continue
            print(f"⚠️ Attribute file does not exist: {attr_path}/{attr_name}.[txt|json]")
            return []
        
        # For sentiment-related attributes, load files for each sentiment label
        else:
            sentiment_data = {}
            for sentiment in ATTRPROMPT_CONFIG['sentiment_labels']:
                # Try different file formats
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
                            break  # Exit the format loop once a file is found
                        except Exception as e:
                            print(f"⚠️ Failed to read sentiment attribute file: {sentiment_file}, error: {e}")
                            continue
                else:
                    # If no file is found, use a default empty list
                    sentiment_data[sentiment] = []
            return sentiment_data
    
    def load_all_attributes(self):
        """Load all attribute data."""
        print("🔄 Loading AttrPrompt attribute data...")
        
        for attr in ATTRPROMPT_CONFIG['attributes']:
            try:
                self.attributes_data[attr] = self.load_attributes(attr)
                print(f"   ✅ {attr}: loaded successfully")
            except Exception as e:
                print(f"   ❌ {attr}: failed to load - {e}")
                self.attributes_data[attr] = []
        
        print("✅ AttrPrompt attribute data loading completed")
    
    def get_attribute_keywords(self, attr_name, sentiment=None):
        """Get keyword list for the given attribute."""
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
    """Attribute compliance score calculator."""
    
    def __init__(self, attr_loader):
        self.attr_loader = attr_loader
    
    def separate_prompt_and_generation(self, completion, prompt=None):
        """Improved prompt/generation separation logic focusing on JSON extraction."""
        try:
            # Method 1: try to extract content from input/output JSON structure
            json_obj = self._find_input_output_json(completion)
            if json_obj and 'input' in json_obj:
                input_text = json_obj['input']
                if isinstance(input_text, str):
                    # Remove "Text: " prefix if present
                    if input_text.startswith("Text: "):
                        return input_text[6:].strip()
                    return input_text.strip()
            
            # Method 2: try to find the "Text: " pattern
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
                    if len(extracted) > 20:  # Ensure the extracted content is meaningful
                        return extracted
            
            # Method 3: try to extract the first meaningful text after JSON
            json_start = completion.find('{')
            if json_start != -1:
                # Find the end position of the JSON block
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
                    # Extract the content after JSON as a fallback
                    remaining = completion[json_end:].strip()
                    if remaining and len(remaining) > 20:
                        # Clean and return the first meaningful line of text
                        lines = remaining.split('\n')
                        for line in lines:
                            line = line.strip()
                            if len(line) > 20 and not line.startswith('#'):
                                return line
            
            # Method 4: simple text cleanup as the last resort
            if len(completion) > 50:
                # Remove obvious code markers
                cleaned = completion
                patterns_to_remove = [
                    r'```[^`]*```',  # Code blocks
                    r'def\s+\w+\([^)]*\):',  # Function definitions
                    r'import\s+\w+',  # Import statements
                    r'#.*$',  # Comments
                ]
                
                for pattern in patterns_to_remove:
                    cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
                
                # Take the first 200 characters as generation content
                cleaned = cleaned.strip()
                if len(cleaned) > 20:
                    return cleaned[:200] + "..." if len(cleaned) > 200 else cleaned
            
            # If all methods fail, return truncated original content
            return completion[:100] + "..." if len(completion) > 100 else completion
            
        except Exception as e:
            print(f"⚠️ Text separation failed: {e}")
            return completion[:100] + "..." if len(completion) > 100 else completion

    def extract_text_from_completion(self, completion, prompt=None):
        """Extract actual review text from completion."""
        try:
            # Import global helper function
            from .novelsum_diversity import separate_prompt_and_generation_global
            
            # First separate the actual generation content
            if prompt is not None:
                generation = self.separate_prompt_and_generation(completion, prompt)
            else:
                generation = separate_prompt_and_generation_global(completion)
            
            # Try to extract from JSON if present
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
        
        # If JSON parsing fails, return the processed text
        return (generation if 'generation' in locals() else completion)

    def extract_sentiment_from_json(self, completion):
        """Extract sentiment label from JSON-formatted completion using input/output fields."""
        try:
            # Method 1: find JSON structure that contains both "input" and "output" fields
            json_obj = self._find_input_output_json(completion)
            if json_obj:
                return self._extract_sentiment_from_parsed_json(json_obj)
            
            # Method 2: if method 1 fails, try traditional JSON extraction
            json_obj = self._find_first_complete_json(completion)
            if json_obj:
                return self._extract_sentiment_from_parsed_json(json_obj)
            
            # Method 3: fallback extraction
            return self._fallback_sentiment_extraction(completion)
                
        except Exception as e:
            print(f"⚠️ JSON parsing failed: {str(e)[:50]}")
            return self._fallback_sentiment_extraction(completion)
    
    def _find_input_output_json(self, text):
        """Find a JSON object that contains both input and output fields."""
        import re
        import json
        
        # Find positions of the "input" keyword
        input_matches = list(re.finditer(r'"input"\s*:', text, re.IGNORECASE))
        
        for input_match in input_matches:
            input_pos = input_match.start()
            
            # Search backwards for the nearest opening brace "{" 
            json_start = -1
            for i in range(input_pos - 1, -1, -1):
                if text[i] == '{':
                    json_start = i
                    break
                elif text[i] == '}':  # If we meet '}', this input is not in the desired JSON
                    break
            
            if json_start == -1:
                continue
            
            # From json_start, try to extract a complete JSON object
            json_obj = self._extract_json_from_position(text, json_start)
            if json_obj and 'input' in json_obj and 'output' in json_obj:
                return json_obj
        
        return None
    
    def _find_first_complete_json(self, text):
        """Find the first complete JSON object in the text."""
        import json
        
        json_start = text.find('{')
        if json_start == -1:
            return None
        
        return self._extract_json_from_position(text, json_start)
    
    def _extract_json_from_position(self, text, start_pos):
        """Extract a complete JSON object starting from the given position."""
        import json
        
        try:
            # Find the matching closing brace
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
            
            # Extract and clean the JSON string
            json_str = text[start_pos:json_end]
            json_str = self._clean_json_string_simple(json_str)
            
            # Try to parse the JSON string
            return json.loads(json_str)
            
        except:
            return None
    
    def _clean_json_string_simple(self, json_str):
        """Simple JSON string cleanup."""
        # Remove control characters but keep basic formatting
        import re
        
        # Remove dangerous control characters
        json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
        
        # Normalize whitespace characters
        json_str = re.sub(r'\s+', ' ', json_str)
        
        return json_str.strip()
    
    def _extract_sentiment_from_parsed_json(self, json_obj):
        """Extract sentiment label from a parsed JSON object."""
        if not isinstance(json_obj, dict):
            return ""
        
        # Prefer checking output-related fields first
        output_keys = ['output', 'sentiment', 'response', 'result', 'answer']
        for key in output_keys:
            if key in json_obj:
                output = json_obj[key]
                if isinstance(output, list):
                    output = output[0] if output else ""
                elif output is not None:
                    output = str(output).strip()
                
                # Check whether this is a valid sentiment label
                if self._is_valid_sentiment(output):
                    return output
        
        # If no standard keys are found, search values that contain sentiment words
        for key, value in json_obj.items():
            if isinstance(value, str):
                value = value.strip()
                if self._is_valid_sentiment(value):
                    return value
        
        return ""
    
    def _is_valid_sentiment(self, text):
        """Check whether the given text is a valid sentiment label."""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        valid_sentiments = [
            'very negative', 'very positive', 'negative', 'positive', 'neutral'
        ]
        
        return text_lower in valid_sentiments
    
    def _fallback_sentiment_extraction(self, completion):
        """Fallback sentiment extraction using simple string matching."""
        # Directly search for sentiment words without complex regular expressions
        sentiments = ['very negative', 'very positive', 'negative', 'positive', 'neutral']
        
        completion_lower = completion.lower()
        for sentiment in sentiments:
            if sentiment in completion_lower:
                return sentiment
        
        return ""

    def calculate_sentiment_consistency(self, completion, target_sentiment):
        """Calculate consistency between extracted and target sentiment labels."""
        extracted_sentiment = self.extract_sentiment_from_json(completion)
        
        # Ensure extracted_sentiment is a string
        if isinstance(extracted_sentiment, list):
            extracted_sentiment = extracted_sentiment[0] if extracted_sentiment else ""
        elif not isinstance(extracted_sentiment, str):
            extracted_sentiment = str(extracted_sentiment) if extracted_sentiment else ""
        
        if not extracted_sentiment:
            return 0.0
        
        # Exact match
        if extracted_sentiment.lower() == target_sentiment.lower():
            return 1.0
        
        # Partial match (e.g., positive vs very positive)
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
        
        return 0.0  # Completely mismatched
    
    def calculate_cuisine_compliance(self, completion, target_cuisine):
        """Calculate cuisine matching score."""
        if not target_cuisine or target_cuisine.lower() == 'default_value':
            return 0.5
        
        completion_lower = completion.lower()
        target_lower = target_cuisine.lower()
        
        # Directly match cuisine name
        if target_lower in completion_lower:
            return 1.0
        
        # Match cuisine-related keywords
        cuisine_keywords = {
            # Existing cuisines
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
            
            # Newly added cuisines
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
        """Calculate length compliance (supports ranges and single target value)."""
        try:
            # Extract actual text from completion
            text_content = self.extract_text_from_completion(completion)
            actual_length = len(text_content.split())
            
            # Debug information
            print(f"DEBUG: target_length type: {type(target_length)}, value: {target_length}")
            print(f"DEBUG: actual_length: {actual_length}")
            
            # Handle different formats of target_length
            if isinstance(target_length, dict):
                # New format: contains range information
                min_length = target_length['min']
                max_length = target_length['max']
                target_value = target_length['target']
                
                # If within the original required range, give full score
                if min_length <= actual_length <= max_length:
                    return 1.0
                
                # If within the extended tolerance range, score based on distance
                extended_min = min_length - tolerance
                extended_max = max_length + tolerance
                
                if extended_min <= actual_length <= extended_max:
                    if actual_length < min_length:
                        # Shorter than required range
                        diff = min_length - actual_length
                        score = 1.0 - (diff / tolerance) * 0.5
                    else:
                        # Longer than required range
                        diff = actual_length - max_length
                        score = 1.0 - (diff / tolerance) * 0.5
                    return max(score, 0.5)
                else:
                    return 0.0
                    
            else:
                # Old format: single target value
                print(f"DEBUG: using old format, target_length: {target_length}")
                target_value = int(target_length) if isinstance(target_length, str) else target_length
                length_diff = abs(actual_length - target_value)
                
                if length_diff <= tolerance:
                    # Within tolerance range, score based on difference
                    score = 1.0 - (length_diff / tolerance) * 0.3
                    return max(score, 0.7)
                elif length_diff <= tolerance * 2:
                    # Outside tolerance range but not too far
                    return 0.5
                else:
                    return 0.0  # Far outside the acceptable range
        except Exception as e:
            print(f"DEBUG: calculate_length_compliance error: {e}")
            return 0.0
    
    def calculate_attribute_keyword_match(self, completion, attr_name, target_value, sentiment=None):
        """Calculate attribute keyword matching score."""
        if not target_value or target_value.lower() == 'default_value':
            return 0.5
        
        completion_lower = completion.lower()
        target_lower = target_value.lower()
        
        # Direct match
        if target_lower in completion_lower:
            return 1.0
        
        # Keyword matching for specific attributes
        score = self._calculate_specific_attribute_match(completion_lower, attr_name, target_lower)
        if score > 0:
            return score
        
        # Get attribute-related keywords (from data files)
        keywords = self.attr_loader.get_attribute_keywords(attr_name, sentiment)
        
        if not keywords:
            return 0.0
        
        # Compute keyword matching score
        matches = 0
        for keyword in keywords:
            if isinstance(keyword, str) and keyword.lower() in completion_lower:
                matches += 1
        
        return min(matches * 0.2, 1.0)
    
    def _calculate_specific_attribute_match(self, completion_lower, attr_name, target_lower):
        """Keyword matching for specific attributes."""
        
        # Style attribute matching
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
        
        # Price range attribute matching
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
        
        # Service quality attribute matching
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
        
        # Atmosphere attribute matching
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
    """Extract all attribute information from the input field of a test sample."""
    attributes = {}
    
    # Extract target sentiment label - patterns adjusted to match actual format
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
            # Normalize sentiment label
            sentiment_mapping = {
                'very negative': 'very negative',
                'negative': 'negative', 
                'neutral': 'neutral',
                'positive': 'positive',
                'very positive': 'very positive'
            }
            attributes['target_sentiment'] = sentiment_mapping.get(sentiment.lower(), sentiment)
            break
    
    # If still not found, try a broader search
    if 'target_sentiment' not in attributes:
        # Search for sentences that contain sentiment words
        sentiment_words = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        for word in sentiment_words:
            if word in input_text.lower():
                attributes['target_sentiment'] = word
                break
    
    # Extract restaurant type (cuisine)
    cuisine_pattern = r"Should be a '([^']+)' restaurant"
    cuisine_match = re.search(cuisine_pattern, input_text)
    if cuisine_match:
        attributes['cuisine'] = cuisine_match.group(1).strip()
    
    # Extract focus/subtopics
    subtopics_pattern = r"Should focus on '([^']+)'"
    subtopics_match = re.search(subtopics_pattern, input_text)
    if subtopics_match:
        attributes['subtopics'] = subtopics_match.group(1).strip()
    
    # Extract length requirement
    length_pattern = r"Should be in length between (\d+) words and (\d+) words"
    length_match = re.search(length_pattern, input_text)
    if length_match:
        min_length, max_length = int(length_match.group(1)), int(length_match.group(2))
        # Store full length range information
        attributes['length'] = {
            'min': min_length,
            'max': max_length,
            'target': (min_length + max_length) // 2
        }
    
    # Extract writing style
    style_pattern = r"The style of the review should be '([^']+)'"
    style_match = re.search(style_pattern, input_text)
    if style_match:
        attributes['style'] = style_match.group(1).strip()
    
    # Extract pricing aspect
    price_pattern = r"The pricing aspect should reflect '([^']+)'"
    price_match = re.search(price_pattern, input_text)
    if price_match:
        attributes['price_range'] = price_match.group(1).strip()
    
    # Extract service quality
    service_pattern = r"The service quality should be described as '([^']+)'"
    service_match = re.search(service_pattern, input_text)
    if service_match:
        attributes['service_quality'] = service_match.group(1).strip()
    
    # Extract atmosphere description
    atmosphere_pattern = r"The atmosphere should be portrayed as '([^']+)'"
    atmosphere_match = re.search(atmosphere_pattern, input_text)
    if atmosphere_match:
        attributes['atmosphere'] = atmosphere_match.group(1).strip()
    
    return attributes

def generate_random_prompt_attributes(attr_loader):
    """Generate a random combination of prompt attributes (fallback when input parsing fails)."""
    if not attr_loader:
        return {}
    
    # Randomly select a sentiment label
    target_sentiment = random.choice(ATTRPROMPT_CONFIG['sentiment_labels'])
    
    # Generate attribute combination
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
                    # For nested dictionaries, randomly select one value
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