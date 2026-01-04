#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing utilities – loading, formatting and preprocessing.
"""
import json
import os
from datasets import Dataset
from attribute_handler import extract_attributes_from_input

def load_synthesis_data(file_path, max_samples=None):
    """Load synthetic data and convert it into a GRPO-compatible format."""
    print(f"Loading synthesis data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    formatted_data = []
    for item in data:
        # Parse the requirements part to obtain the true target attributes
        if 'instruction' in item and 'input' in item and 'output' in item:
            try:
                # Extract the requirements text from the input
                requirements_text = item['input']
                attributes = extract_attributes_from_input(requirements_text)
                true_target_sentiment = attributes.get('target_sentiment', 'neutral')
                
                # Parse the review text from the output (for reference only, not a training target)
                output_json = json.loads(item['output'])
                review_text = output_json.get('input', '').replace('Text: ', '')
                generated_sentiment = output_json.get('output', '')  # Sentiment predicted by GPT (may be noisy)
                
                # Build GRPO-style messages (only prompt is needed; no reference answer)
                user_message = f"{item['instruction']}\n\n{item['input']}"
                
                formatted_item = {
                    'messages': [
                        {'role': 'user', 'content': user_message}
                        # Note: GRPO does not require an assistant message; the model generates it
                    ],
                    # Key information: use the true target sentiment from the requirements
                    'sentiment': true_target_sentiment,  # True target sentiment
                    'generated_sentiment': generated_sentiment,  # Sentiment from GPT output (for comparison)
                    'review_text': review_text,  # Generated review text
                    'original_input': item['input']  # Preserve original requirements text for attribute extraction
                }
                formatted_data.append(formatted_item)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Skipping malformed sample: {e}")
                continue
                
        # Compatibility branch: handle the legacy messages-format data
        elif 'messages' in item and len(item['messages']) >= 1:
            # For the legacy messages format, extract the true target sentiment
            user_content = item['messages'][0].get('content', '')
            attributes = extract_attributes_from_input(user_content)
            
            formatted_item = {
                'messages': item['messages'],
                'sentiment': attributes.get('target_sentiment', 'neutral'),
                'original_input': user_content
            }
            
            # Preserve other useful fields if present
            for key in ['review_text', 'generated_sentiment']:
                if key in item:
                    formatted_item[key] = item[key]
                    
            formatted_data.append(formatted_item)
        else:
            print(f"⚠️ Skipping unsupported data format: {list(item.keys())}")
    
    print(f"Loaded {len(formatted_data)} samples")
    return formatted_data

def create_sentiment_grouped_dataset(data, batch_size=4):
    """Create a dataset grouped by sentiment labels to optimize small-batch training."""
    print(f"🔄 Reorganizing dataset by sentiment label (target batch size: {batch_size})...")
    
    # Group by sentiment label
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
            # Handle unknown sentiment labels by mapping them to neutral
            print(f"⚠️ Unknown sentiment label: {sentiment}, falling back to 'neutral'")
            sentiment_groups['neutral'].append(item)
                
    # Print grouping statistics
    print("📊 Sentiment group statistics:")
    for sentiment, items in sentiment_groups.items():
        print(f"   {sentiment}: {len(items)} samples")
    
    # Reorganize data so that each batch is as label-homogeneous as possible
    reorganized_data = []
    
    # Create batches for each sentiment label
    for sentiment, items in sentiment_groups.items():
        if items:  # Only handle non-empty groups
            # Split samples of a specific sentiment into batches
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                reorganized_data.extend(batch_items)
                print(f"   Added batch for {sentiment}: {len(batch_items)} samples")
    
    print(f"✅ Dataset reorganization completed: {len(reorganized_data)} samples")
    print(f"🎯 Expected number of batches: {len(reorganized_data) // batch_size}")
    
    return reorganized_data

def prepare_grpo_dataset(data_list):
    """Prepare the dataset used for GRPO training."""
    print("🔄 Preparing GRPO dataset...")
    
    # Add a prompt field for GRPO and remove messages to avoid conflicts
    for item in data_list:
        if 'messages' in item and len(item['messages']) > 0:
            # Use the first user message content as the prompt
            user_message = item['messages'][0]['content']
            item['prompt'] = user_message
            # Remove messages field to avoid conflicts with GRPO format
            del item['messages']
    
    dataset = Dataset.from_list(data_list)
    print(f"✅ GRPO dataset ready with {len(dataset)} samples")
    
    return dataset

def validate_dataset_format(dataset):
    """Validate the dataset format used for training."""
    print("🔍 Validating dataset format...")
    sample_item = dataset[0]
    print(f"   Fields: {list(sample_item.keys())}")
    if 'prompt' in sample_item:
        print(f"   Example prompt length: {len(sample_item['prompt'])}")
    if 'sentiment' in sample_item:
        print(f"   Example sentiment label: {sample_item['sentiment']}")
    print("✅ Dataset format validation completed")
    
    return True

def create_optimized_dataset(file_path, max_samples=None, batch_size=4):
    """Full pipeline to create an optimized GRPO training dataset."""
    # Load raw synthetic data
    data = load_synthesis_data(file_path, max_samples)
    print(f"🔄 Loaded raw data: {len(data)} samples")
    
    # Apply sentiment-grouping optimization
    print("🧠 Applying sentiment-grouping optimization...")
    grouped_data = create_sentiment_grouped_dataset(data, batch_size)
    print(f"✅ Sentiment grouping completed, optimized sample count: {len(grouped_data)}")
    
    # Prepare GRPO dataset
    dataset = prepare_grpo_dataset(grouped_data)
    
    # Validate final format
    validate_dataset_format(dataset)
    
    return dataset, grouped_data