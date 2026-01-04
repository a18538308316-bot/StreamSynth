#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNLI evaluation script for Qwen with LoRA adapter.
"""
import os
import json
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict

class MNLIEvaluatorQwen:
    def __init__(self, base_model_path: str, adapter_path: str):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
    def load_model(self):
        print(f"Loading base model: {self.base_model_path}")
        print(f"Loading MNLI adapter: {self.adapter_path}")
        if self.adapter_path.startswith('./'):
            self.adapter_path = os.path.abspath(self.adapter_path)
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
        print("MNLI Qwen model loaded successfully!")
    def generate_response(self, prompt: str) -> str:
        """Generate a single response (keeps behavior same as Llama variant)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses for a list of prompts by calling single-sample generator."""
        responses = []
        for prompt in prompts:
            responses.append(self.generate_response(prompt))
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
            return response[first_brace:end_pos]
        return response
    def extract_json_or_label(self, response: str) -> tuple:
        json_result = self.extract_json(response)
        nli_labels = ['entailment', 'contradiction', 'neutral']
        valid_label = None
        has_json = False
        if json_result != response and '{' in json_result and '}' in json_result:
            try:
                parsed = json.loads(json_result)
                label = parsed.get('output', '').lower()
                if label in nli_labels:
                    valid_label = label
                    has_json = True
            except:
                pass
        if not valid_label:
            response_lower = response.lower()
            for label in nli_labels:
                if label in response_lower:
                    valid_label = label
                    break
        return has_json, valid_label
    def evaluate_mnli_dataset(self, test_data: List[Dict], batch_size: int = 8) -> List[Dict]:
        results = []
        total_samples = len(test_data)
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_samples = test_data[batch_start:batch_end]
            prompts = []
            for sample in batch_samples:
                instruction = sample.get('instruction', '')
                user_input = sample.get('input', '')
                full_prompt = f"{instruction}\n\n{user_input}\n\nResponse:"
                prompts.append(full_prompt)

            # keep same batching behavior as original Llama script
            if batch_size == 1:
                raw_responses = [self.generate_response(prompts[0])]
            else:
                raw_responses = self.generate_responses(prompts)
            for i, raw_response in enumerate(raw_responses):
                sample = batch_samples[i]
                expected_output = sample.get('output', '')
                has_json, valid_label = self.extract_json_or_label(raw_response)
                results.append({
                    'sample_id': batch_start + i,
                    'prompt': prompts[i],
                    'response': raw_response,
                    'expected_output': expected_output,
                    'has_valid_json': has_json,
                    'has_valid_nli_label': valid_label is not None,
                    'nli_label': valid_label
                })
        return results

def main():
    parser = argparse.ArgumentParser(description="MNLI Qwen Model Evaluation")
    parser.add_argument("--base_model", type=str,
                      default="Qwen/Qwen2.5-7B-Instruct",
                      help="Base model name or path (Qwen2.5-7B-Instruct by default)")
    parser.add_argument("--adapter_path", type=str, default="./mnli_model_output_qwen")
    parser.add_argument("--test_dataset", type=str,
                      default="./MNLI_test_1.json",
                      help="Path to MNLI test dataset (JSON file)")
    parser.add_argument("--output_file", type=str, default="./mnli_evaluation_results_qwen.json")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    with open(args.test_dataset, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    if len(test_data) > args.sample_size:
        test_data = random.sample(test_data, args.sample_size)
    evaluator = MNLIEvaluatorQwen(args.base_model, args.adapter_path)
    evaluator.load_model()
    results = evaluator.evaluate_mnli_dataset(test_data, batch_size=args.batch_size)
    valid_json_count = sum(1 for r in results if r['has_valid_json'])
    valid_label_count = sum(1 for r in results if r['has_valid_nli_label'])
    valid_response_count = sum(1 for r in results if r['has_valid_json'] or r['has_valid_nli_label'])
    valid_json_rate = valid_json_count / len(results)
    valid_label_rate = valid_label_count / len(results)
    valid_response_rate = valid_response_count / len(results)
    output_data = {
        "task": "MNLI Natural Language Inference Qwen",
        "model_type": "qwen",
        "total_samples": len(results),
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_rate,
        "valid_nli_label_count": valid_label_count,
        "valid_nli_label_rate": valid_label_rate,
        "valid_response_count": valid_response_count,
        "valid_response_rate": valid_response_rate,
        "results": results
    }
    print(f"\nSaving MNLI Qwen results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print("\n=== MNLI Qwen Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Valid JSON responses: {valid_json_count} ({valid_json_rate:.2%})")
    print(f"Valid NLI labels: {valid_label_count} ({valid_label_rate:.2%})")
    print(f"Valid responses (JSON or label): {valid_response_count} ({valid_response_rate:.2%})")
    print("MNLI Qwen evaluation completed successfully!")

if __name__ == "__main__":
    main()
