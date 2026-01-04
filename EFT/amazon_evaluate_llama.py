#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concise Amazon model evaluation script.

Focuses on core functionality: load model, generate responses,
extract JSON, and save results.
"""

import os
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict

class SimpleEvaluator:
    def __init__(self, base_model_path: str, adapter_path: str):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load base model and LoRA adapter."""
        print(f"Loading base model: {self.base_model_path}")
        print(f"Loading adapter: {self.adapter_path}")

        # Handle relative adapter path
        if self.adapter_path.startswith('./'):
            self.adapter_path = os.path.abspath(self.adapter_path)
            print(f"Converted to absolute path: {self.adapter_path}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load PEFT adapter
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str) -> str:
        """Generate a model response for a single prompt."""
        return self.generate_responses([prompt])[0]
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate model responses in batch."""
        responses = []

        for prompt in prompts:
            # Use the same method as in the simple test script
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)
            
            # Generation parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,
                    temperature=0.3,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            
            # Extract only the newly generated part
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def extract_json(self, response: str) -> str:
        """Extract the first complete JSON object from the response."""
        # Find the first '{' and its matching '}'
        first_brace = response.find('{')
        if first_brace == -1:
            # No JSON found
            return response

        # From the first '{', find the corresponding closing '}'
        brace_count = 0
        end_pos = len(response)
        
        for i, char in enumerate(response[first_brace:], first_brace):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        if brace_count == 0:
            # Found a complete JSON object
            json_part = response[first_brace:end_pos]
            return json_part
        else:
            # Incomplete JSON, return raw response
            return response

    def evaluate_dataset(self, test_data: List[Dict], batch_size: int = 1) -> List[Dict]:
        """Evaluate the Amazon dataset with optional batching.

        Amazon test data format:
        {
            "instruction": "...",
            "input": "Requirements: ...",
            "output": "{\"input\": \"Text: ...\", \"output\": \"positive\"}"
        }
        """
        results = []
        total_samples = len(test_data)

        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_samples = test_data[batch_start:batch_end]

            print(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start+1}-{batch_end}/{total_samples}")

            # Prepare prompts for the current batch
            batch_prompts = []
            for sample in batch_samples:
                instruction = sample.get('instruction', '')
                user_input = sample.get('input', '')
                # Amazon format: instruction + input + a blank line
                full_prompt = f"{instruction}\n\n{user_input}"
                batch_prompts.append(full_prompt)

            # Generate responses for the current batch
            if batch_size == 1:
                # Single-sample processing
                raw_responses = [self.generate_response(batch_prompts[0])]
            else:
                # Batched processing
                raw_responses = self.generate_responses(batch_prompts)

            # Post-process batch results
            for i, (sample, raw_response) in enumerate(zip(batch_samples, raw_responses)):
                sample_id = batch_start + i
                expected_output = sample.get('output', '')

                # Extract JSON
                extracted_json = self.extract_json(raw_response)

                # Check whether a JSON-like structure is present
                has_json = ('{' in extracted_json and '}' in extracted_json and 
                           '"input"' in extracted_json and '"output"' in extracted_json)

                # Additionally validate whether the sentiment label is valid for Amazon
                try:
                    import json as json_module
                    parsed_response = json_module.loads(extracted_json)
                    sentiment_label = parsed_response.get('output', '')
                    valid_sentiment = sentiment_label in ['very negative', 'negative', 'neutral', 'positive', 'very positive']
                except:
                    valid_sentiment = False
                
                # Save result entry
                result = {
                    "sample_id": sample_id,
                    "prompt": batch_prompts[i],
                    "response": extracted_json,
                    "expected_output": expected_output,
                    "has_valid_json": has_json,
                    "has_valid_sentiment": valid_sentiment
                }

                results.append(result)

                if has_json and valid_sentiment:
                    print(f"  ✅ Sample {sample_id+1}: Valid JSON + sentiment")
                elif has_json:
                    print(f"  ⚠️ Sample {sample_id+1}: Valid JSON but invalid sentiment")
                else:
                    print(f"  ❌ Sample {sample_id+1}: No valid JSON found")

        return results

def main():
    parser = argparse.ArgumentParser(description="Simple Model Evaluation")
    parser.add_argument("--base_model", type=str, 
                      default="meta-llama/Meta-Llama-3-8B-Instruct",
                      help="Base model name or path (Meta-Llama-3-8B-Instruct by default)")
    parser.add_argument("--adapter_path", type=str, 
                      default="./synthesis_model_output_amazon",
                      help="Path to the adapter")
    parser.add_argument("--test_dataset", type=str,
                      default="./amazon_test_1.json",
                      help="Path to Amazon test dataset (JSON file)")
    parser.add_argument("--output_file", type=str, 
                      default="./simple_evaluation_results_amazon.json",
                      help="Output file path")
    parser.add_argument("--sample_size", type=int, default=100,
                      help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for model inference")
    
    args = parser.parse_args()
    
    print("=== Simple Model Evaluation ===")
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Sample size: {args.sample_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {args.output_file}")

    # Load test data
    print("\nLoading test dataset...")
    with open(args.test_dataset, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} samples")

    # Optional sampling
    if len(test_data) > args.sample_size:
        import random
        test_data = random.sample(test_data, args.sample_size)
        print(f"Sampled {args.sample_size} samples for evaluation")

    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = SimpleEvaluator(args.base_model, args.adapter_path)
    evaluator.load_model()

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate_dataset(test_data, batch_size=args.batch_size)

    # Compute summary statistics
    valid_json_count = sum(1 for r in results if r['has_valid_json'])
    valid_sentiment_count = sum(1 for r in results if r.get('has_valid_sentiment', False))
    valid_json_rate = valid_json_count / len(results)
    valid_sentiment_rate = valid_sentiment_count / len(results)

    # Save results
    output_data = {
        "total_samples": len(results),
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_rate,
        "valid_sentiment_count": valid_sentiment_count, 
        "valid_sentiment_rate": valid_sentiment_rate,
        "results": results
    }
    
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n=== Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Valid JSON responses: {valid_json_count}")
    print(f"Valid JSON rate: {valid_json_rate:.2%}")
    print(f"Valid sentiment responses: {valid_sentiment_count}")
    print(f"Valid sentiment rate: {valid_sentiment_rate:.2%}")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
