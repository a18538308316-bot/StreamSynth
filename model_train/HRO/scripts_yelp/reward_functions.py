#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward function module - contains all sample-level and batch-level reward functions.
"""
import torch
import numpy as np
from datetime import datetime
# Remove novelsum-related imports but keep necessary helper functions

# Import batch diversity reward module
try:
    from .batch_diversity_reward import reward_batch_diversity, initialize_batch_diversity_calculator
    print("✅ Batch diversity reward module imported successfully")
except ImportError as e:
    print(f"⚠️ Failed to import batch diversity reward module: {e}")
    reward_batch_diversity = None
    initialize_batch_diversity_calculator = None

# Global variables (will be injected from the main module)
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
training_visualizer = None  # Training visualizer

# Sample-level reward weights (default configuration)
SAMPLE_REWARDS_CONFIG = {
    'sentiment_consistency_weight': 0.25,  # Sentiment label consistency
    'attribute_compliance_weight': 0.15,   # Attribute compliance
    'length_compliance_weight': 0.10,      # Length compliance
}

# Batch-level reward weights (default configuration)
BATCH_REWARDS_CONFIG = {
    'batch_diversity_weight': 0.30,  # Batch diversity reward based on local density
}

# Optimized configuration (will be dynamically updated)
CURRENT_SAMPLE_REWARDS_CONFIG = SAMPLE_REWARDS_CONFIG.copy()
CURRENT_BATCH_REWARDS_CONFIG = BATCH_REWARDS_CONFIG.copy()

# =============================================================================
# Utility functions (necessary helpers migrated from the novelsum module)
# =============================================================================

def extract_text_content_global(text):
    """Global helper to extract clean text content (simplified version)."""
    if not text:
        return ""
    
    # Simple text cleaning
    text = str(text).strip()
    
    # Remove extra whitespace characters
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text

def separate_prompt_and_generation_global(completion, prompt=""):
    """Global helper to separate prompt and generation (simplified version)."""
    if not completion:
        return "", ""
    
    completion = str(completion).strip()
    prompt = str(prompt).strip()
    
    # If a prompt is provided, try to strip it from the completion
    if prompt and completion.startswith(prompt):
        generation = completion[len(prompt):].strip()
        return prompt, generation
    
    # If there is no prompt or separation fails, return the whole completion as generation
    return prompt, completion

def get_current_batch_attributes(step, batch_size):
    """Get attribute information for the current batch based on training step."""
    global training_data_global, current_batch_index
    
    if not training_data_global:
        print("⚠️ Training data not initialized, using default attributes")
        return {'target_sentiment': 'neutral', 'cuisine': 'american', 'length': 200}
    
    # Use cyclic access to avoid going out of range
    total_samples = len(training_data_global)
    
    # Use modulo to ensure the index stays in range
    start_idx = (step * batch_size) % total_samples
    
    # If we are looping over data again, log a message but do not fail
    if step * batch_size >= total_samples:
        epoch_num = (step * batch_size) // total_samples + 1
        if step % 50 == 0:  # Print every 50 steps to avoid log flooding
            print(f"🔄 Training enters epoch {epoch_num}, step {step}, using cyclic data access")
    
    # Get the original_input of the first sample in the current batch
    try:
        sample = training_data_global[start_idx]
        if 'original_input' in sample:
            from attribute_handler import extract_attributes_from_input
            attributes = extract_attributes_from_input(sample['original_input'])
            print(f"🎯 Step {step}: Extract attributes from data - {attributes.get('target_sentiment', 'unknown')}")
            return attributes
        else:
            print(f"⚠️ Sample {start_idx} is missing 'original_input' field")
            return {'target_sentiment': 'neutral', 'cuisine': 'american', 'length': 200}
    except Exception as e:
        print(f"⚠️ Error while getting batch attributes: {e}")
        return {'target_sentiment': 'neutral', 'cuisine': 'american', 'length': 200}

def extract_attributes_from_current_prompts(prompts):
    """Extract target attributes directly from the current prompts."""
    if not prompts or len(prompts) == 0:
        return {}
    
    # Use the first prompt as representative (prompts in the same batch should share the same requirements)
    first_prompt = prompts[0]
    
    # Extract attributes from prompt
    from attribute_handler import extract_attributes_from_input
    
    # Directly extract attributes from the prompt text
    attributes = extract_attributes_from_input(first_prompt)
    
    return attributes

def update_current_prompt_attributes_from_prompts(prompts):
    """Update attributes from current prompts (the preferred method)."""
    global current_prompt_attributes
    
    if prompts:
        current_prompt_attributes = extract_attributes_from_current_prompts(prompts)
        print(f"🎯 Extract attributes from prompt: sentiment={current_prompt_attributes.get('target_sentiment', 'unknown')}, cuisine={current_prompt_attributes.get('cuisine', 'unknown')}")
    else:
        print("⚠️ No prompts provided, cannot extract attributes")

def update_current_prompt_attributes(step=None, batch_size=None):
    """Update prompt attributes for the current batch (deprecated; prefer extracting from prompts)."""
    global current_prompt_attributes, current_batch_index, batch_size_global
    
    if step is not None and batch_size is not None:
        current_prompt_attributes = get_current_batch_attributes(step, batch_size)
        print(f"🔄 Updating prompt attributes from historical data (may be inaccurate): {current_prompt_attributes}")
    elif step is not None:
        current_prompt_attributes = get_current_batch_attributes(step, batch_size_global)
        print("🔄 Updating prompt attributes using default batch_size")
    else:
        print("⚠️ Step not provided, cannot update prompt attributes")

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
    
    # Forward to training visualizer (if exists)
    if training_visualizer is not None and hasattr(training_visualizer, 'record_reward_data'):
        training_visualizer.record_reward_data(step, reward_type, rewards, completions)

# =============================================================================
# Sample-level reward functions
# =============================================================================

def reward_sentiment_consistency_batch(completions, **kwargs):
    """Reward for sentiment label consistency."""
    global compliance_calculator, current_training_step, current_prompt_attributes
    rewards = []
    print(f"🎯 Step {current_training_step} - Evaluating sentiment consistency reward ({len(completions)} samples)")
    
    # Debug: inspect kwargs
    if current_training_step <= 2:  # Show detailed info for the first 3 steps
        print(f"🔍 Debug kwargs keys: {list(kwargs.keys())}")
        if 'prompts' in kwargs:
            print(f"🔍 Prompts available: {len(kwargs['prompts'])}")
            
        # Show full prompt and completion
        print("=" * 80)
        print(f"📝 STEP {current_training_step} - Full PROMPT and COMPLETION comparison")
        print("=" * 80)
        
        for i, completion in enumerate(completions[:2]):  # Show only the first 2 samples
            print(f"\n🔸 Sample {i+1}:")
            print("📥 Full PROMPT:")
            if 'prompts' in kwargs and i < len(kwargs['prompts']):
                prompt = kwargs['prompts'][i]
                print(f"'{prompt}'")
            else:
                print("  [PROMPT not found]")
            
            print("\n📤 Full COMPLETION:")
            print(f"'{completion}'")
            
            print("\n🔍 Separated generation content:")
            try:
                if 'prompts' in kwargs and i < len(kwargs['prompts']):
                    prompt = kwargs['prompts'][i]
                    _, separated_generation = separate_prompt_and_generation_global(completion, prompt)
                else:
                    _, separated_generation = separate_prompt_and_generation_global(completion, "")
                print(f"'{separated_generation[:300]}...'")
            except Exception as e:
                print(f"   ❌ Failed to separate: {e}")
                separated_generation = completion
            
            print("-" * 60)
        print("=" * 80)
        
    # Extract target attributes from prompts (preferred way)
    if 'prompts' in kwargs:
        update_current_prompt_attributes_from_prompts(kwargs['prompts'])
    else:
        # If there are no prompts, try the old (less accurate) way
        if not current_prompt_attributes:
            update_current_prompt_attributes()
    
    target_sentiment = current_prompt_attributes.get('target_sentiment', 'neutral')
    
    for i, completion in enumerate(completions):
        score = compliance_calculator.calculate_sentiment_consistency(completion, target_sentiment)
        
        # Positive reinforcement: high reward for full match, neutral reward otherwise (avoid negative scores)
        if score >= 1.0:  # Perfect match
            final_reward = 0.5
        elif score >= 0.5:  # Partial match
            final_reward = 0.2
        else:  # Not matching, but do not punish (neutral)
            final_reward = 0.0
        rewards.append(final_reward)
        
        if i < 2:
            extracted_sentiment = compliance_calculator.extract_sentiment_from_json(completion)
            # Show processed generation content instead of raw completion
            generated_text = extract_text_content_global(completion)
            print(f"   Sample {i+1}: target_sentiment={target_sentiment}, extracted_sentiment={extracted_sentiment}, score={score:.2f}, reward={final_reward:.4f}")
            print(f"   Generated content: {generated_text[:100]}...")
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # Remove step from kwargs to avoid passing it twice
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('step', None)
    
    log_reward_details(current_training_step, "sentiment_consistency", completions, rewards_tensor, **kwargs_copy)
    return rewards_tensor

def reward_attribute_compliance_batch(completions, **kwargs):
    """Reward for attribute compliance."""
    global compliance_calculator, current_training_step, current_prompt_attributes
    rewards = []
    print(f"🔍 Step {current_training_step} - Evaluating attribute compliance reward ({len(completions)} samples)")
    
    # Ensure we have current prompt attributes
    if not current_prompt_attributes:
        update_current_prompt_attributes()
    
    for i, completion in enumerate(completions):
        total_score = 0.0
        attribute_count = 0
        
        # Evaluate cuisine attribute
        if 'cuisine' in current_prompt_attributes:
            cuisine_score = compliance_calculator.calculate_cuisine_compliance(
                completion, current_prompt_attributes['cuisine']
            )
            total_score += cuisine_score
            attribute_count += 1
        
        # Evaluate other attributes
        for attr_name in ['style', 'price_range', 'service_quality', 'atmosphere']:
            if attr_name in current_prompt_attributes:
                attr_score = compliance_calculator.calculate_attribute_keyword_match(
                    completion, 
                    attr_name, 
                    current_prompt_attributes[attr_name],
                    current_prompt_attributes.get('target_sentiment')
                )
                total_score += attr_score * 0.5  # Lower weight for other attributes
                attribute_count += 0.5
        
            # Compute average score
        if attribute_count > 0:
            avg_score = total_score / attribute_count
        else:
            avg_score = 0.5
        
        # Positive reinforcement: give reward for compliance, no penalty for non-compliance
        if avg_score >= 0.8:  # Highly compliant
            final_reward = 0.25
        elif avg_score >= 0.6:  # Good compliance
            final_reward = 0.15
        elif avg_score >= 0.4:  # Basic compliance
            final_reward = 0.1
        elif avg_score >= 0.2:  # Partial compliance
            final_reward = 0.05
        else:  # Not compliant, but do not punish
            final_reward = 0.0
        rewards.append(final_reward)
        
        if i < 2:
            cuisine = current_prompt_attributes.get('cuisine', 'N/A')
            print(f"   Sample {i+1}: target_cuisine={cuisine}, avg_attribute_score={avg_score:.2f}, reward={final_reward:.4f}")
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # Remove step from kwargs to avoid passing it twice
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('step', None)
    
    log_reward_details(current_training_step, "attribute_compliance", completions, rewards_tensor, **kwargs_copy)
    return rewards_tensor

def reward_length_compliance_batch(completions, **kwargs):
    """Reward for complying with length requirements."""
    global compliance_calculator, current_training_step, current_prompt_attributes
    rewards = []
    print(f"📏 Step {current_training_step} - Evaluating length compliance reward ({len(completions)} samples)")
    
    # Ensure we have current prompt attributes
    if not current_prompt_attributes:
        update_current_prompt_attributes()
    
    target_length = current_prompt_attributes.get('length', 200)
    
    for i, completion in enumerate(completions):
        score = compliance_calculator.calculate_length_compliance(completion, target_length, tolerance=25)
        
        # Positive reinforcement: reward if within range, no penalty otherwise
        if score >= 1.0:  # Perfectly meets length requirement
            final_reward = 0.2
        elif score >= 0.8:  # Mostly meets
            final_reward = 0.1
        elif score >= 0.5:  # Acceptable range
            final_reward = 0.05
        else:  # Not compliant, but no penalty (avoid conflict with other requirements)
            final_reward = 0.0
        rewards.append(final_reward)
        
        if i < 2:
            # Compute actual length for display
            text_content = extract_text_content_global(completion)
            actual_length = len(text_content.split())
            
            # Show target length info
            if isinstance(target_length, dict):
                length_info = f"target_range={target_length['min']}-{target_length['max']}"
            else:
                length_info = f"target_length={target_length}"
            
            print(f"   Sample {i+1}: {length_info}, actual_length={actual_length}, score={score:.2f}, reward={final_reward:.4f}")
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # Remove step from kwargs to avoid passing it twice
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('step', None)
    
    log_reward_details(current_training_step, "length_compliance", completions, rewards_tensor, **kwargs_copy)
    return rewards_tensor

# =============================================================================
# Generation quality reward functions
# =============================================================================

def extract_clean_review_global(generated_text, prompt=""):
    """Extract a clean review from generated text (keep JSON format)."""
    import re
    import json
    
    # Remove prompt part
    if prompt and generated_text.startswith(prompt):
        content = generated_text[len(prompt):].strip()
    else:
        content = generated_text.strip()
    
    # Try to extract a complete JSON
    json_pattern = r'\{[^{}]*"input"\s*:\s*"[^"]*"[^{}]*"output"\s*:\s*"[^"]*"[^{}]*\}'
    json_match = re.search(json_pattern, content)
    if json_match:
        try:
            # Validate that JSON format is correct
            json_str = json_match.group(0)
            parsed_json = json.loads(json_str)
            if 'input' in parsed_json and 'output' in parsed_json:
                return json_str  # Return the complete JSON
        except json.JSONDecodeError:
            pass
    
    # If no full JSON is found, try to extract the input field
    input_match = re.search(r'"input"\s*:\s*"([^"]*)"', content)
    if input_match:
        input_content = input_match.group(1)
        # Try to extract the output field
        output_match = re.search(r'"output"\s*:\s*"([^"]*)"', content)
        if output_match:
            output_content = output_match.group(1)
            # Reconstruct JSON
            return f'{{"input": "{input_content}", "output": "{output_content}"}}'
        else:
            # Only input found, add a default output
            return f'{{"input": "{input_content}", "output": "unknown"}}'
    
    # If everything fails, try extracting long content inside quotes
    quote_matches = re.findall(r'"([^"]{20,})"', content)
    if quote_matches:
        # Choose the longest match as the main content and wrap as JSON
        longest_content = max(quote_matches, key=len)
        return f'{{"input": "Text: {longest_content}", "output": "unknown"}}'
    
    # Final fallback: remove instruction-like text and wrap as JSON
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
    
    # Clean up extra whitespace and special characters
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'^["\'\s]+|["\'\s]+$', '', content)
    
    if len(content) > 10:
        return f'{{"input": "Text: {content}", "output": "unknown"}}'
    else:
        return '{"input": "Text: Invalid generation", "output": "unknown"}'

def calculate_generation_quality_score_global(generated_text, prompt=""):
    """Enhanced generation quality scoring (strongly penalize bad generations)."""
    import json
    import re
    
    clean_content = extract_clean_review_global(generated_text, prompt)
    
    # 🔥 Start from a low base score that must be earned upwards
    quality_score = 0.1
    
    # 🚨 Strongly penalize programming/code content and bad patterns
    bad_patterns = [
        'import ', 'def ', 'class ', 'function', '```', 'python', 'code', 'script',
        '# ', 'return ', 'print(', 'if __name__', 'from ', 'pipeline', 'random.',
        'gen_', 'tokenizer', 'model_output', 'transformers', 'torch', 'numpy',
        'def synthesize', 'def gen_review', 'def get_random'
    ]
    
    bad_count = sum(1 for pattern in bad_patterns if pattern.lower() in generated_text.lower())
    if bad_count > 0:
        quality_score -= min(0.8, bad_count * 0.3)  # Strong penalty per bad pattern
        
    # 🚨 Strongly penalize repeated characters and meaningless content
    repeat_matches = re.findall(r'(.)\1{10,}', generated_text)
    if repeat_matches:
        quality_score -= 0.6
    
    # 🚨 Penalize instruction-like text
    instruction_phrases = [
        'here is an example', 'here\'s an example', 'do not add', 'note that you', 
        'template', 'format', 'please see', 'answer:', 'step 1:', 'step 2:',
        'feel free', 'let me know', 'best regards'
    ]
    bad_instructions = sum(1 for phrase in instruction_phrases if phrase.lower() in generated_text.lower())
    if bad_instructions > 0:
        quality_score -= min(0.6, bad_instructions * 0.2)
    
    # ✅ Reward correct JSON format
    try:
        parsed_json = json.loads(clean_content)
        if 'input' in parsed_json and 'output' in parsed_json:
            quality_score += 0.4  # Large bonus for correct JSON format
            
            # Check input field format
            input_content = parsed_json.get('input', '')
            if input_content.startswith('Text: ') and len(input_content) > 20:
                quality_score += 0.3  # Bonus for correct input format
                
                # Extract actual review content
                review_text = input_content[6:]  # Remove "Text: "
                word_count = len(review_text.split())
                
                # Check length of review
                if 20 <= word_count <= 300:
                    quality_score += 0.2
                elif word_count < 10:
                    quality_score -= 0.3
                elif word_count > 500:
                    quality_score -= 0.1
            else:
                quality_score -= 0.2  # Input format is not correct
            
            # Check whether output field is a valid sentiment label
            output_content = parsed_json.get('output', '')
            valid_sentiments = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
            if output_content in valid_sentiments:
                quality_score += 0.1
            elif output_content == 'unknown':
                quality_score -= 0.1  # Smaller penalty for unknown label
            else:
                quality_score -= 0.2  # Invalid label
        else:
            quality_score -= 0.3  # JSON missing required fields
    except json.JSONDecodeError:
                quality_score -= 0.3  # Heavy penalty for invalid JSON
    
    # 3. Check for excessive formatting characters or noise
    noise_patterns = [r'```', r'"""', r'\{[^}]*\}[^}]*\{', r'\\n\\n\\n+']
    for pattern in noise_patterns:
        if re.search(pattern, generated_text):
            quality_score -= 0.1
    
    # 4. Check if there are obvious code/template leftovers
    code_indicators = ['def ', 'import ', 'class ', '# ', '// ', '/* ']
    if any(indicator in generated_text for indicator in code_indicators):
        quality_score -= 0.3
    
    return max(0.0, min(1.0, quality_score))

def reward_generation_quality_batch(completions, **kwargs):
    """Reward function for overall generation quality."""
    global current_training_step
    rewards = []
    prompts = kwargs.get('prompts', [''] * len(completions))
    
    print(f"🎨 Step {current_training_step} - Evaluating generation quality reward ({len(completions)} samples)")
    
    for i, completion in enumerate(completions):
        prompt = prompts[i] if i < len(prompts) else ""
        quality_score = calculate_generation_quality_score_global(completion, prompt)
        
        # Positive reinforcement: reward high quality, do not punish low quality
        if quality_score >= 0.9:  # Excellent quality
            reward = 0.3
        elif quality_score >= 0.7:  # Good quality
            reward = 0.2
        elif quality_score >= 0.5:  # Acceptable quality
            reward = 0.1
        elif quality_score >= 0.3:  # Average quality, small reward
            reward = 0.05
        else:  # Low quality, but no penalty (avoid confusing the model)
            reward = 0.0
        rewards.append(reward)
        
        if i < 2:  # Show detailed info for the first 2 samples
            clean_content = extract_clean_review_global(completion, prompt)
            print(f"   Sample {i+1}: quality_score={quality_score:.2f}, reward={reward:.4f}")
            print(f"   Cleaned content: {clean_content[:100]}...")
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # Remove step from kwargs to avoid passing it twice
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('step', None)
    
    log_reward_details(current_training_step, "generation_quality", completions, rewards_tensor, **kwargs_copy)
    return rewards_tensor

# =============================================================================
# Batch-level reward functions
# =============================================================================

def reward_yelp_semantic_diversity_batch(completions, **kwargs):
    """Semantic diversity reward for restaurant reviews (NovelSum-inspired, supports dynamic reference data)."""
    global novelsum_calculator, current_training_step
    
    print(f"🍽️ Step {current_training_step} - Evaluating restaurant review semantic diversity reward (batch size: {len(completions)})")
    
    try:
        # Extract actual text content from completions
        texts = [extract_text_content_global(completion) for completion in completions]
        texts = [text for text in texts if text and len(text) > 10]
        
        if len(texts) < 2:
            print("   ⚠️ Less than 2 valid texts, returning base reward")
            return torch.zeros(len(completions), dtype=torch.float32)
        
        # Use NovelSum to compute diversity
        novelsum_score = novelsum_calculator.calculate_novelsum_score(
            texts,
            density_power=0.5,
            distance_power=1.0,
            neighbors=min(10, len(texts))
        )
        
        # Compute restaurant-specific diversity
        print("🔍 DEBUG: Start computing restaurant diversity...")
        restaurant_diversity = calculate_restaurant_specific_diversity(texts)
        print(f"🔍 DEBUG: Restaurant diversity computed: {restaurant_diversity} (type: {type(restaurant_diversity)})")
        
        # Combine the two diversity metrics
        print("🔍 DEBUG: Start combining diversity metrics...")
        print(f"🔍 DEBUG: novelsum_score = {novelsum_score} (type: {type(novelsum_score)})")
        print(f"🔍 DEBUG: restaurant_diversity = {restaurant_diversity} (type: {type(restaurant_diversity)})")
        
        combined_diversity = (novelsum_score * 0.6 + restaurant_diversity * 0.4)
        print(f"🔍 DEBUG: initial combined_diversity = {combined_diversity} (type: {type(combined_diversity)})")
        
        combined_diversity = float(combined_diversity)  # Ensure scalar
        print(f"🔍 DEBUG: converted combined_diversity = {combined_diversity} (type: {type(combined_diversity)})")
        
        # Distribute batch-level reward to each sample
        print("🔍 DEBUG: Start computing batch reward...")
        batch_reward = (combined_diversity - 0.5) * 0.6  # Normalize to [-0.3, 0.3]
        print(f"🔍 DEBUG: initial batch_reward = {batch_reward} (type: {type(batch_reward)})")
        
        batch_reward = float(batch_reward)  # Ensure scalar
        print(f"🔍 DEBUG: converted batch_reward = {batch_reward} (type: {type(batch_reward)})")
        
        rewards = torch.full((len(completions),), batch_reward, dtype=torch.float32)
        print(f"🔍 DEBUG: rewards tensor created: shape={rewards.shape}")
        
        # Dynamically update reference pool (relaxed conditions to encourage learning)
        print("🔍 DEBUG: Check whether to update reference pool...")
        print(f"🔍 DEBUG: hasattr(novelsum_calculator, 'add_training_samples') = {hasattr(novelsum_calculator, 'add_training_samples')}")
        print(f"🔍 DEBUG: batch_reward = {batch_reward}, batch_reward > -0.1 = {batch_reward > -0.1}")
        
        if hasattr(novelsum_calculator, 'add_training_samples') and batch_reward > -0.1:
            try:
                # Compute per-sample quality score (based on diversity and length)
                quality_scores = []
                for text in texts:
                    length_score = min(1.0, len(text.split()) / 150.0)  # Normalized by length
                    
                    # Safely compute individual_novelty
                    if hasattr(novelsum_calculator, 'reference_manager'):
                        individual_novelty = novelsum_calculator.reference_manager.compute_novelty_score(text)
                        # Ensure scalar
                        if isinstance(individual_novelty, torch.Tensor):
                            individual_novelty = float(individual_novelty.item())
                        else:
                            individual_novelty = float(individual_novelty)
                    else:
                        individual_novelty = 0.5
                    
                    quality = (individual_novelty * 0.7 + length_score * 0.3)
                    quality_scores.append(float(quality))
                
                # Add high-quality samples to reference pool
                added_count = novelsum_calculator.add_training_samples(texts, quality_scores)
                if added_count > 0:
                    print(f"   📊 Added {added_count} high-quality samples to dynamic reference pool")
                    
            except Exception as e:
                print(f"   ⚠️ Failed to update dynamic reference pool: {e}")
                import traceback
                print(f"   Detailed error: {traceback.format_exc()}")
        
        print(f"   NovelSum score: {novelsum_score:.3f}, restaurant diversity: {restaurant_diversity:.3f}")
        print(f"   Combined diversity: {combined_diversity:.3f}, batch reward: {batch_reward:.4f}")
        
        # Periodically print reference pool statistics
        if current_training_step % 20 == 0 and hasattr(novelsum_calculator, 'get_reference_statistics'):
            stats = novelsum_calculator.get_reference_statistics()
            print(f"   📈 Reference pool status: {stats.get('total_count', 0)} samples (original: {stats.get('original_count', 0)}, dynamic: {stats.get('dynamic_count', 0)})")
        
        # Remove step from kwargs to avoid passing it twice
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('step', None)
        log_reward_details(current_training_step, "yelp_semantic_diversity", completions, rewards, **kwargs_copy)
        return rewards
        
    except Exception as e:
        print(f"   ⚠️ Diversity computation failed: {e}")
        return torch.zeros(len(completions), dtype=torch.float32)

def reward_inter_sample_diversity_batch(completions, **kwargs):
    """Reward for inter-sample diversity within a batch."""
    global current_training_step
    
    print(f"🔄 Step {current_training_step} - Evaluating inter-sample diversity reward (batch size: {len(completions)})")
    
    try:
        texts = [extract_text_content_global(completion) for completion in completions]
        texts = [text for text in texts if text and len(text) > 10]
        
        if len(texts) < 2:
            return torch.zeros(len(completions), dtype=torch.float32)
        
        # Use simple Jaccard distance to compute diversity
        def jaccard_similarity(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        # Compute diversity for all sample pairs
        diversities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = jaccard_similarity(texts[i], texts[j])
                diversity = 1.0 - similarity
                diversities.append(diversity)
        
        if not diversities:
            return torch.zeros(len(completions), dtype=torch.float32)
        
        avg_diversity = np.mean(diversities)
        avg_diversity = float(avg_diversity)  # Ensure scalar
        
        # Normalize reward into [-0.2, 0.2]
        batch_reward = (avg_diversity - 0.5) * 0.4
        batch_reward = float(batch_reward)  # Ensure scalar
        rewards = torch.full((len(completions),), batch_reward, dtype=torch.float32)
        
        print(f"   Mean Jaccard diversity: {avg_diversity:.3f}, batch reward: {batch_reward:.4f}")
        
        # Remove step from kwargs to avoid passing it twice
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('step', None)
        log_reward_details(current_training_step, "inter_sample_diversity", completions, rewards, **kwargs_copy)
        return rewards
        
    except Exception as e:
        print(f"   ⚠️ Batch diversity computation failed: {e}")
        return torch.zeros(len(completions), dtype=torch.float32)

# =============================================================================
# Combined reward functions
# =============================================================================

def create_weighted_reward_functions():
    """Create weighted reward functions (sample-level + batch-level)."""
    
    def weighted_sentiment_consistency(completions, **kwargs):
        """Weighted sentiment consistency reward function."""
        global reward_call_counter, current_training_step
        
        # Update global counter and training step
        reward_call_counter += 1
        current_training_step = reward_call_counter // 5  # Every 5 reward calls correspond to one training step (we currently have 5 reward functions)
        
        # Update prompt attributes for current batch
        update_current_prompt_attributes(current_training_step)
        
        base_rewards = reward_sentiment_consistency_batch(completions, **kwargs)
        return base_rewards  # Return unweighted base reward
    
    def weighted_attribute_compliance(completions, **kwargs):
        """Attribute compliance reward function (returns unweighted values)."""
        base_rewards = reward_attribute_compliance_batch(completions, **kwargs)
        return base_rewards  # Return unweighted base reward
    
    def weighted_length_compliance(completions, **kwargs):
        """Length compliance reward function (returns unweighted values)."""
        base_rewards = reward_length_compliance_batch(completions, **kwargs)
        return base_rewards  # Return unweighted base reward
    
    def weighted_generation_quality(completions, **kwargs):
        """Generation quality reward function (unweighted)."""
        base_rewards = reward_generation_quality_batch(completions, **kwargs)
        return base_rewards  # Return unweighted base reward
    
    def weighted_batch_diversity(completions, **kwargs):
        """Batch-level diversity reward function."""
        if reward_batch_diversity is None:
            print("⚠️ Batch diversity reward function not available, returning neutral reward")
            return [0.0] * len(completions)
        
        base_rewards = reward_batch_diversity(completions, **kwargs)
        return base_rewards  # Return unweighted base reward
    
    # Set function names to be consistent with original reward function identifiers
    weighted_sentiment_consistency.__name__ = "reward_sentiment_consistency"
    weighted_attribute_compliance.__name__ = "reward_attribute_compliance"
    weighted_length_compliance.__name__ = "reward_length_compliance"
    weighted_generation_quality.__name__ = "reward_generation_quality"
    weighted_batch_diversity.__name__ = "reward_batch_diversity"
    
    return [
        weighted_sentiment_consistency,    # Sample-level: sentiment label consistency
        weighted_attribute_compliance,     # Sample-level: attribute compliance  
        weighted_length_compliance,        # Sample-level: length compliance
        weighted_generation_quality,       # Sample-level: generation quality control
        weighted_batch_diversity,          # Batch-level: diversity reward based on local density
    ]

def save_training_logs(output_dir):
    """Save training reward logs to disk."""
    global reward_logs
    
    if reward_logs:
        logs_file = f"{output_dir}/reward_logs.json"
        import json
        with open(logs_file, 'w', encoding='utf-8') as f:
            json.dump(reward_logs, f, indent=2, ensure_ascii=False)
        print(f"📊 Reward logs saved to: {logs_file}")

def initialize_reward_globals(td_global, bs_global, rc, nc, al, cc, optimized_sample_config=None, optimized_batch_config=None, embedding_model_path=None):
    """Initialize global variables for reward functions (sample + batch level)."""
    global training_data_global, batch_size_global, reward_calculator, novelsum_calculator, attr_loader, compliance_calculator
    global CURRENT_SAMPLE_REWARDS_CONFIG, CURRENT_BATCH_REWARDS_CONFIG
    
    training_data_global = td_global
    batch_size_global = bs_global
    reward_calculator = rc
    novelsum_calculator = nc  # Kept for compatibility but not actively used
    attr_loader = al
    compliance_calculator = cc
    
    # Update configuration if optimized config is provided
    if optimized_sample_config:
        CURRENT_SAMPLE_REWARDS_CONFIG.update(optimized_sample_config)
        print(f"✅ Using optimized sample reward config: {CURRENT_SAMPLE_REWARDS_CONFIG}")
    
    # Update batch config
    if optimized_batch_config:
        CURRENT_BATCH_REWARDS_CONFIG.update(optimized_batch_config)
        print(f"✅ Using optimized batch reward config: {CURRENT_BATCH_REWARDS_CONFIG}")
    
    # Initialize batch diversity calculator
    if embedding_model_path and initialize_batch_diversity_calculator:
        try:
            initialize_batch_diversity_calculator(embedding_model_path, device='cuda', k_penalty=2.0)
            print("✅ Batch diversity calculator initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize batch diversity calculator: {e}")
    
    print("✅ Reward function module globals initialized (sample + batch level)")

def set_training_visualizer(visualizer):
    """Set the training visualizer instance."""
    global training_visualizer
    training_visualizer = visualizer
    print("✅ Training visualizer has been set")