#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concise evaluation script for Qwen.
Only adapts model/tokenizer loading for Qwen specifics (trust_remote_code=True, pad_token fallback); other logic stays the same as the original script.
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
        print(f"Loading base model: {self.base_model_path}")
        print(f"Loading adapter: {self.adapter_path}")
        if self.adapter_path.startswith('./'):
            self.adapter_path = os.path.abspath(self.adapter_path)
            print(f"Converted to absolute path: {self.adapter_path}")
        # Qwen: trust_remote_code=True for both model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str) -> str:
        return self.generate_responses([prompt])[0]
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)
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
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
        return responses
    
    def extract_json(self, response: str) -> str:
        first_brace = response.find('{')
        if first_brace == -1:
            return response
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
            json_part = response[first_brace:end_pos]
            return json_part
        else:
            return response
    
    def evaluate_dataset(self, test_data: List[Dict], batch_size: int = 1) -> List[Dict]:
        results = []
        total_samples = len(test_data)
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_samples = test_data[batch_start:batch_end]
            print(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start+1}-{batch_end}/{total_samples}")
            batch_prompts = []
            for sample in batch_samples:
                instruction = sample.get('instruction', '')
                user_input = sample.get('input', '')
                full_prompt = f"{instruction}\n\n{user_input}\n\nResponse:"
                batch_prompts.append(full_prompt)
            if batch_size == 1:
                raw_responses = [self.generate_response(batch_prompts[0])]
            else:
                raw_responses = self.generate_responses(batch_prompts)
            for i, (sample, raw_response) in enumerate(zip(batch_samples, raw_responses)):
                sample_id = batch_start + i
                expected_output = sample.get('output', '')
                extracted_json = self.extract_json(raw_response)
                has_json = ('{' in extracted_json and '}' in extracted_json and 
                           '"input"' in extracted_json and '"output"' in extracted_json)
                result = {
                    "sample_id": sample_id,
                    "prompt": batch_prompts[i],
                    "response": extracted_json,
                    "expected_output": expected_output,
                    "has_valid_json": has_json
                }
                results.append(result)
        return results

def main():
    parser = argparse.ArgumentParser(description="Simple Model Evaluation (Qwen)")
    parser.add_argument("--base_model", type=str,
                      default="Qwen/Qwen2.5-7B-Instruct",
                      help="Base model name or path (Qwen2.5-7B-Instruct by default)")
    parser.add_argument("--adapter_path", type=str, 
                      default="./synthesis_model_output_improved_qwen",
                      help="Path to the adapter")
    parser.add_argument("--test_dataset", type=str,
                      default="./test_data_1_100.json",
                      help="Path to test dataset (JSON file)")
    parser.add_argument("--output_file", type=str, 
                      default="./simple_evaluation_results_qwen.json",
                      help="Output file path")
    parser.add_argument("--sample_size", type=int, default=100,
                      help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for model inference")
    args = parser.parse_args()
    print("=== Simple Model Evaluation (Qwen) ===")
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Sample size: {args.sample_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {args.output_file}")
    with open(args.test_dataset, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} samples")
    if len(test_data) > args.sample_size:
        import random
        test_data = random.sample(test_data, args.sample_size)
        print(f"Sampled {args.sample_size} samples for evaluation")
    evaluator = SimpleEvaluator(args.base_model, args.adapter_path)
    evaluator.load_model()
    results = evaluator.evaluate_dataset(test_data, batch_size=args.batch_size)
    valid_json_count = sum(1 for r in results if r['has_valid_json'])
    valid_json_rate = valid_json_count / len(results) if len(results)>0 else 0.0
    output_data = {
        "total_samples": len(results),
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_rate,
        "results": results
    }
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print("\n=== Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Valid JSON responses: {valid_json_count}")
    print(f"Valid JSON rate: {valid_json_rate:.2%}")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
