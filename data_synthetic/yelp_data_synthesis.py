import json
import transformers
import torch
import pandas as pd
import random
import os
import argparse
import re
import numpy as np
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading

def load_attributes(attr_name, dataset='yelp', model='gpt-3.5-turbo', classes=None):
    """Load attribute metadata for Yelp synthesis."""
    base_path = f"data/{dataset}/prompt/{model}/{attr_name}"
    
    # For general attributes (e.g., cuisine, length, style), load a single file
    general_attrs = ['cuisine', 'length', 'style']
    if attr_name in general_attrs:
        file_path = f"{base_path}/{attr_name}.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, using default values")
            return ['default_value']
    
    # For label-specific attributes, load one file per label
    else:
        attr_dict = {}
        for class_name in classes:
            file_path = f"{base_path}/{class_name.replace(' ', '_')}.jsonl"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    attr_dict[class_name] = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                print(f"Warning: {file_path} not found, using default for {class_name}")
                attr_dict[class_name] = ['default_value']
        return attr_dict

def generate_attribute_combination(attr_dict, target_label):
    """Generate a random attribute combination for the target label."""
    combination = {}
    
    for attr_name, attr_values in attr_dict.items():
        if isinstance(attr_values, dict):
            # Label-specific attributes: use values for the target label
            if target_label in attr_values:
                combination[attr_name] = random.choice(attr_values[target_label])
            else:
                # If target label not found, use 'neutral' or the first available value
                if 'neutral' in attr_values:
                    combination[attr_name] = random.choice(attr_values['neutral'])
                else:
                    combination[attr_name] = random.choice(list(attr_values.values())[0])
        else:
            # General attributes: select randomly
            combination[attr_name] = random.choice(attr_values)
    
    return combination

# Prompt 1: enhanced prompt with dynamic attribute control
D1_quality_prompt = """
You are a data synthesis assistant. 
Your task is to generate exactly ONE new Yelp review based on the given examples.

The sentiment label is always chosen from the following set:
["very negative", "negative", "neutral", "positive", "very positive"]

- "very negative": extreme dissatisfaction, anger, strong complaints.
- "negative": clear dissatisfaction, but not extreme.
- "neutral": mixed experience, balanced between good and bad.
- "positive": clear satisfaction, but not extreme.
- "very positive": extremely satisfied, strong praise, enthusiastic tone.

Requirements:
1. Generate ONE new review only. Do not copy sentences from the input.
2. Preserve the sentiment label exactly as provided. Never change or reinterpret it.
3. Follow the specific requirements below:
   - The overall review should be {target_label}
   - Should be a '{cuisine}' restaurant
   - Should focus on '{subtopics}'
   - Should be in length between {length} words and {length_max} words
   - The style of the review should be '{style}'
   - The pricing aspect should reflect '{price_range}'
   - The service quality should be described as '{service_quality}'
   - The atmosphere should be portrayed as '{atmosphere}'
4. Write in a natural human Yelp style, informal and conversational. Small imperfections (typos, fragmented sentences) are allowed.

Input examples with their sentiment labels:

Example 1 (very negative):
{example_very_negative}

Example 2 (negative):
{example_negative_1}

Example 3 (negative):
{example_negative_2}

Example 4 (neutral):
{example_neutral_1}

Example 5 (neutral):
{example_neutral_2}

Example 6 (positive):
{example_positive_1}

Example 7 (positive):
{example_positive_2}

Example 8 (very positive):
{example_very_positive}

Target sentiment for generation: {target_label}

Generate exactly one review with {target_label} sentiment following all the requirements above.

Output ONLY in this exact JSON format:
{{
  "input": "Text: <write your new review here>",
  "output": "{target_label}"
}}
"""

# Define prompt mapping - keep only enhanced prompt version
PROMPT_MAPPING = {
    "prompt": D1_quality_prompt
}

def load_yelp_attributes():
    """Load Yelp attribute metadata used for prompting."""
    print("Loading Yelp attributes...")
    
    # Define attribute list and labels
    attributes = ["cuisine", "subtopics", "style", "price_range", "service_quality", "atmosphere", "length"]
    labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    
    attr_dict = {}
    
    for attr in attributes:
        print(f"  Loading {attr}...")
        attr_dict[attr] = load_attributes(attr_name=attr, dataset='yelp', model='gpt-3.5-turbo', classes=labels)
    
    print("Attributes loaded successfully!")
    return attr_dict

def create_model_interface(use_api, model_path=None, api_key=None, base_url=None, model_name=None):
    """Create a model interface supporting local models and API calls."""
    if use_api:
        # Use remote API
        if not all([api_key, base_url, model_name]):
            raise ValueError("API mode requires api_key, base_url, and model_name")
        
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        return {"type": "api", "client": client, "model": model_name}
    else:
        # Use local model
        if not model_path:
            raise ValueError("Local mode requires model_path")
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        # Set pad_token and pad_token_id
        pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
        
        return {"type": "local", "pipeline": pipeline}

def load_yelp_data():
    """Load Yelp training data."""
    try:
        with open('yelp_train_llama_factory.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"Successfully loaded Yelp training data with {len(train_data)} samples")
        return train_data
    except Exception as e:
        print(f"Failed to load Yelp data: {e}")
        return []

def load_yelp_labels():
    """Load Yelp sentiment label list."""
    try:
        with open('yelp_labels.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"Successfully loaded {len(labels)} Yelp labels: {labels}")
        return labels
    except Exception as e:
        print(f"Failed to load Yelp labels: {e}")
        return ["very negative", "negative", "neutral", "positive", "very positive"]

def get_prompt_template(quality_type="prompt"):
    """Return the prompt template used for generation."""
    return PROMPT_MAPPING["prompt"]

def group_data_by_labels(yelp_data):
    """Group Yelp data by sentiment label."""
    grouped_data = {
        "very negative": [],
        "negative": [],
        "neutral": [],
        "positive": [],
        "very positive": []
    }
    
    for item in yelp_data:
        label = item.get('output', '').strip()
        if label in grouped_data:
            grouped_data[label].append(item)
    
    # Log number of samples per label
    for label, items in grouped_data.items():
        print(f"Label '{label}': {len(items)} samples")
    
    return grouped_data

def sample_examples_for_prompt(grouped_data):
    """Randomly sample 8 examples per label distribution for prompting.

    Distribution:
    - very negative: 1 sample
    - negative: 2 samples
    - neutral: 2 samples
    - positive: 2 samples
    - very positive: 1 sample
    """
    examples = []
    
    # Define required number of samples per label
    sample_distribution = {
        "very negative": 1,
        "negative": 2,
        "neutral": 2,
        "positive": 2,
        "very positive": 1
    }
    
    for label, count in sample_distribution.items():
        if label in grouped_data and len(grouped_data[label]) >= count:
            sampled_items = random.sample(grouped_data[label], count)
            for item in sampled_items:
                examples.append({
                    "label": label,
                    "input": item.get('input', ''),
                    "output": item.get('output', '')
                })
        else:
            print(f"Warning: Not enough samples for label '{label}'. Required: {count}, Available: {len(grouped_data.get(label, []))}")
    
    return examples

def create_prompt_with_quality(attr_dict, examples, target_label):
    """Create an enhanced prompt with dynamic attribute control."""
    template = get_prompt_template()
    
    # Generate a dynamic attribute combination for the target label
    attr_combination = generate_attribute_combination(attr_dict, target_label)
    
    # Prepare formatted example strings using correct variable names
    example_texts = {}
    
    # Group examples by label
    data_by_label = {}
    for item in examples:
        label = item['output']
        if label not in data_by_label:
            data_by_label[label] = []
        data_by_label[label].append(item)
    
    # Sample examples per label
    def get_example_text(label, fallback_text="This restaurant was okay."):
        if label in data_by_label and len(data_by_label[label]) > 0:
            sample = random.choice(data_by_label[label])
            input_text = sample['input']
            if input_text.startswith("Text: "):
                return input_text[6:]
            return input_text
        return fallback_text
    
    example_texts['example_very_negative'] = get_example_text('very negative', "This place was terrible!")
    example_texts['example_negative_1'] = get_example_text('negative', "Not great experience.")
    example_texts['example_negative_2'] = get_example_text('negative', "Disappointing visit.")
    example_texts['example_neutral_1'] = get_example_text('neutral', "It was okay.")
    example_texts['example_neutral_2'] = get_example_text('neutral', "Mixed experience.")
    example_texts['example_positive_1'] = get_example_text('positive', "Good experience.")
    example_texts['example_positive_2'] = get_example_text('positive', "Enjoyed it.")
    example_texts['example_very_positive'] = get_example_text('very positive', "Amazing place!")
    
    # Attach attribute values
    example_texts['target_label'] = target_label
    example_texts['cuisine'] = attr_combination['cuisine']
    example_texts['subtopics'] = attr_combination['subtopics']
    example_texts['style'] = attr_combination['style']
    example_texts['price_range'] = attr_combination['price_range']
    example_texts['service_quality'] = attr_combination['service_quality']
    example_texts['atmosphere'] = attr_combination['atmosphere']
    
    # Handle length-related attributes
    length_value = attr_combination['length']
    try:
        length_num = int(length_value)
        example_texts['length'] = str(length_num)
        example_texts['length_max'] = str(length_num + 50)
    except ValueError:
        example_texts['length'] = "100"
        example_texts['length_max'] = "150"
    
    return template.format(**example_texts), attr_combination

def generate_with_model(model_interface, prompt, max_tokens, temperature):
    """Generate content with either an API model or a local model."""
    if model_interface["type"] == "api":
        print(f"Making API call to {model_interface['model']}...")
        # Remote API call
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = model_interface["client"].chat.completions.create(
                model=model_interface["model"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            result = response.choices[0].message.content
            if result is None:
                result = ""
                print("✗ API returned None content")
            elif len(result.strip()) == 0:
                print("✗ API returned empty content")
            else:
                print(f"✓ API call successful (length: {len(result)})")
            
            return result
        except Exception as e:
            print(f"✗ API call failed: {e}")
            raise e
    
    else:
        # Local model call
        output = model_interface["pipeline"](
            prompt,
            max_new_tokens=max_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )
        
        if isinstance(output, list) and len(output) > 0:
            return output[0]['generated_text']
        else:
            return str(output)

def parse_synthesis_output(output_text, valid_labels):
    """Parse model output and extract new input/output pairs."""
    # If output is empty or too short, preserve raw output as input
    if not output_text or len(output_text.strip()) < 10:
        print(f"Warning: Empty or too short output (len={len(output_text)}), preserving as raw input")
        return {
            "input": f"RAW_OUTPUT_PARSE_FAILED: {output_text}",
            "output": "neutral"
        }
    
    # Handle potential <think>...</think> wrapper from some models
    if "<think>" in output_text and "</think>" in output_text:
        print(f"Debug: Found <think> tags in output, extracting content after </think>")
        # Find the closing </think> tag and extract the content after it
        think_end = output_text.find("</think>")
        if think_end != -1:
            # Extract everything after </think>
            content_after_think = output_text[think_end + 8:].strip()  # 8 is the length of "</think>"
            if content_after_think:
                print(f"Debug: Extracted content after </think>: {content_after_think[:100]}...")
                output_text = content_after_think
            else:
                print(f"Warning: No content found after </think> tag")
        else:
            print(f"Warning: Found <think> but not </think> tag")
    
    try:
        # Try to parse JSON-formatted output
        if "{" in output_text and "}" in output_text:
            # Extract JSON segment
            start_idx = output_text.find("{")
            end_idx = output_text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = output_text[start_idx:end_idx + 1]
                try:
                    parsed_json = json.loads(json_str)
                    input_text = parsed_json.get("input", "")
                    output_label = parsed_json.get("output", "")
                    
                    # Clean input text
                    if input_text.startswith("Text: "):
                        input_text = input_text[6:]
                    
                    # Validate that we parsed meaningful content
                    if input_text and len(input_text.strip()) > 10:
                        # Validate label
                        if output_label not in valid_labels:
                            print(f"Warning: Invalid label '{output_label}', using 'neutral' as fallback")
                            output_label = "neutral"
                        
                        return {
                            "input": input_text.strip(),
                            "output": output_label
                        }
                    else:
                        print("Warning: Parsed input text is too short or empty, preserving raw output")
                        return {
                            "input": f"RAW_OUTPUT_JSON_PARSED_BUT_EMPTY: {output_text}",
                            "output": "neutral"
                        }
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode failed: {e}, trying manual parsing")
                    pass
        
        # If JSON parsing fails, try manual extraction
        lines = output_text.split('\n')
        input_text = ""
        output_label = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('"input":') or line.startswith('input:'):
                input_text = line.split(':', 1)[1].strip().strip('"').strip(',')
                if input_text.startswith("Text: "):
                    input_text = input_text[6:]
            elif line.startswith('"output":') or line.startswith('output:'):
                output_label = line.split(':', 1)[1].strip().strip('"').strip(',')
        
        # Validate manual parsing result
        if input_text and len(input_text.strip()) > 10:
            # Validate label again
            if output_label not in valid_labels:
                print(f"Warning: Invalid label '{output_label}', using 'neutral' as fallback")
                output_label = "neutral"
            
            return {
                "input": input_text.strip(),
                "output": output_label if output_label else "neutral"
            }
        else:
            # Manual parsing also failed, preserve raw output
            print("Warning: Manual parsing failed, preserving raw output")
            return {
                "input": f"RAW_OUTPUT_MANUAL_PARSE_FAILED: {output_text}",
                "output": "neutral"
            }
        
    except Exception as e:
        print(f"Error parsing output: {e}, preserving raw output")
        return {
            "input": f"RAW_OUTPUT_EXCEPTION: {output_text}",
            "output": "neutral"
        }

def generate_single_sample(args_tuple):
    """Generate a single synthetic sample (supports multi-threaded use)."""
    (sample_id, target_label, model_interface, attr_dict, grouped_data, 
     yelp_labels, max_tokens, temperature) = args_tuple
    
    try:
        print(f"Thread processing sample {sample_id} (target: {target_label})")
        
        # Randomly sample 8 examples for the prompt
        examples = sample_examples_for_prompt(grouped_data)
        
        # Build the enhanced prompt
        prompt, attr_combination = create_prompt_with_quality(attr_dict, examples, target_label)
        
        # Call model for generation
        generated_text = generate_with_model(model_interface, prompt, max_tokens, temperature)
        
        # Parse model output
        parsed_result = parse_synthesis_output(generated_text, yelp_labels)
        
        # Ensure generated label matches target label
        if parsed_result["output"] != target_label:
            print(f"Warning: Generated label '{parsed_result['output']}' doesn't match target '{target_label}'")
            # Force using target label to keep label distribution balanced
            parsed_result["output"] = target_label
        
        # Build final synthesized record
        synthesis_record = {
            "instruction": "Classify the sentiment of the given text. Choose: very negative, negative, neutral, positive, very positive",
            "input": parsed_result["input"],
            "output": parsed_result["output"],
            "metadata": {
                "synthesis_id": sample_id,
                "timestamp": datetime.now().isoformat(),
                "method": "prompt",
                "target_label": target_label,
                "examples_used": len(examples),
                "attribute_combination": attr_combination,
                "raw_output": generated_text
            }
        }
        
        print(f"✓ Sample {sample_id} completed (generated: {parsed_result['output']})")
        return synthesis_record
        
    except Exception as e:
        print(f"✗ Error processing sample {sample_id}: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Yelp Data Synthesis with prompt Enhancement')
    parser.add_argument('--use_api', action='store_true', help='Use API instead of local model')
    parser.add_argument('--model_path', type=str, default='/path/to/llama3.1_70b', help='Local model path')
    parser.add_argument('--api_key', type=str, help='API key for OpenAI-compatible service')
    parser.add_argument('--base_url', type=str, help='Base URL for OpenAI-compatible service')
    parser.add_argument('--model_name', type=str, help='Model name for API calls')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--samples_per_label', type=int, default=1000, help='Number of samples per sentiment label')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index for processing (for resuming)')
    parser.add_argument('--max_workers', type=int, default=5, help='Number of concurrent threads for API calls')
    parser.add_argument('--save_interval', type=int, default=10, help='Save progress every N samples')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Add thread lock for API calls
    save_lock = threading.Lock()
    
    # Change to your project root if needed before running synthesis
    # os.chdir('/path/to/your/project_root/data_synthetic')
    
    print("="*50)
    print("Yelp Data Synthesis with prompt Enhancement")
    print("="*50)
    
    # Create output directory
    output_dir = 'synthesis_output_prompt_api'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model interface
    try:
        model_interface = create_model_interface(
            use_api=args.use_api,
            model_path=args.model_path,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name
        )
        print(f"Model interface created successfully. Type: {model_interface['type']}")
    except Exception as e:
        print(f"Failed to create model interface: {e}")
        return
    
    # Load Yelp attribute data
    print("Loading Yelp attributes...")
    try:
        attr_dict = load_yelp_attributes()
        print(f"✓ Successfully loaded {len(attr_dict)} attribute categories")
    except Exception as e:
        print(f"✗ Failed to load attributes: {e}")
        return
    
    # Load data
    print("Loading Yelp training data...")
    yelp_data = load_yelp_data()
    yelp_labels = load_yelp_labels()
    
    if not yelp_data:
        print("Unable to load data, exiting")
        return
    
    print(f"✓ Loaded {len(yelp_data)} training samples")
    
    # Group data by sentiment label
    print("Grouping data by sentiment labels...")
    grouped_data = group_data_by_labels(yelp_data)
    
    # Verify each label has enough data
    min_samples_needed = max(2, 1)  # Require at least 2 samples (for negative and positive each)
    for label, samples in grouped_data.items():
        if len(samples) < min_samples_needed:
            print(f"Error: Label '{label}' has only {len(samples)} samples, need at least {min_samples_needed}")
            return
    
    # Create output file name
    output_file = os.path.join(output_dir, f'yelp_synthesized_prompt.json')
    
    # Check for existing output files
    synthesized_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                synthesized_data = json.load(f)
            print(f"Found existing output file with {len(synthesized_data)} synthesized samples")
        except Exception as e:
            print(f"Failed to load existing output file: {e}")
            synthesized_data = []
    
    # Count synthesized samples per label
    label_counts = {}
    for item in synthesized_data:
        label = item.get('output', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Current generation status:")
    for label in yelp_labels:
        current_count = label_counts.get(label, 0)
        print(f"  {label}: {current_count}/{args.samples_per_label}")
    
    # Build generation plan: target number of samples per label
    generation_plan = []
    for label in yelp_labels:
        current_count = label_counts.get(label, 0)
        remaining = args.samples_per_label - current_count
        if remaining > 0:
            generation_plan.extend([label] * remaining)
    
    # Shuffle generation order randomly
    random.shuffle(generation_plan)
    
    print(f"Generation plan: {len(generation_plan)} samples to generate")
    label_plan_counts = {}
    for label in generation_plan:
        label_plan_counts[label] = label_plan_counts.get(label, 0) + 1
    for label, count in label_plan_counts.items():
        print(f"  Will generate {count} samples for '{label}'")
    
    start_idx = max(args.start_idx, len(synthesized_data))
    print(f"Starting generation from index {start_idx}")
    
    # Prepare task list for multi-threaded processing
    tasks = []
    for i, target_label in enumerate(generation_plan):
        sample_id = start_idx + i + 1
        task_args = (sample_id, target_label, model_interface, attr_dict, grouped_data, 
                    yelp_labels, args.max_tokens, args.temperature)
        tasks.append(task_args)
    
    print(f"Using {args.max_workers} concurrent threads for API calls...")
    
    # Use thread pool for concurrent API calls
    completed_count = 0
    if args.use_api and len(tasks) > 0:
        # Use multi-threading only in API mode
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(generate_single_sample, task) for task in tasks]
            
            # Handle completed tasks
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        with save_lock:  # Use lock to protect shared data
                            synthesized_data.append(result)
                            completed_count += 1
                            
                            # Periodically save progress
                            if completed_count % args.save_interval == 0:
                                with open(output_file, 'w', encoding='utf-8') as f:
                                    json.dump(synthesized_data, f, ensure_ascii=False, indent=2)
                                print(f"Progress saved: {completed_count}/{len(tasks)} samples completed")
                
                except Exception as e:
                    print(f"Error in thread execution: {e}")
                    
    else:
        # Use single thread for local model (avoid GPU resource conflicts)
        for i, task_args in enumerate(tasks):
            try:
                result = generate_single_sample(task_args)
                if result is not None:
                    synthesized_data.append(result)
                    completed_count += 1
                    
                    # Periodically save progress
                    if completed_count % args.save_interval == 0:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(synthesized_data, f, ensure_ascii=False, indent=2)
                        print(f"Progress saved: {completed_count}/{len(tasks)} samples completed")
            except Exception as e:
                print(f"Error processing sample {i + 1}: {e}")
                continue
    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(synthesized_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSynthesis completed! Generated {len(synthesized_data)} synthesis samples")
    print(f"Output file saved at: {output_file}")
    
    # Compute final generation statistics
    final_label_counts = {}
    for item in synthesized_data:
        label = item.get('output', 'unknown')
        final_label_counts[label] = final_label_counts.get(label, 0) + 1
    
    print("Final generation statistics:")
    for label in yelp_labels:
        count = final_label_counts.get(label, 0)
        print(f"  {label}: {count} samples")
    
    # Generate statistics report
    stats_file = os.path.join(output_dir, f'yelp_synthesis_stats_prompt.json')
    stats = {
        "total_synthesized": len(synthesized_data),
        "source_dataset": "Yelp Reviews",
        "source_data_count": len(yelp_data),
        "synthesis_method": "prompt",
        "samples_per_label_target": args.samples_per_label,
        "final_label_distribution": final_label_counts,
        "synthesis_completion_time": datetime.now().isoformat(),
        "model_config": {
            "use_api": args.use_api,
            "model_path": args.model_path if not args.use_api else None,
            "model_name": args.model_name if args.use_api else None,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature
        },
        "valid_labels": yelp_labels,
        "attribute_categories": list(attr_dict.keys())
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Statistics saved at: {stats_file}")

if __name__ == "__main__":
    main()
