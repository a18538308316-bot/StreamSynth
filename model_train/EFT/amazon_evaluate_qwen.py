#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concise Amazon evaluation script for Qwen with LoRA adapter.

Batch generation is optimized: do_sample=False, max_new_tokens=128.
"""

import os
import json
import argparse
import random
import sys
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict

class SimpleEvaluatorQwen:
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()
        print('Model loaded')

    def generate_responses(self, prompts: List[str]) -> List[str]:
        # Use single-sample generation consistent with the Llama simple evaluator
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=4096
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.3,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            responses.append(self.tokenizer.decode(generated_tokens, skip_special_tokens=True))
        return responses

    def extract_json(self, response: str) -> str:
        first_brace = response.find('{')
        if first_brace == -1:
            return response
        brace_count = 0
        end_pos = len(response)
        for i, ch in enumerate(response[first_brace:], first_brace):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i+1
                    break
        if brace_count == 0:
            return response[first_brace:end_pos]
        return response

    def evaluate_dataset(self, test_data: List[Dict], batch_size: int = 8) -> List[Dict]:
        results = []
        total = len(test_data)
        for start in range(0, total, batch_size):
            end = min(start+batch_size, total)
            batch = test_data[start:end]
            prompts = []
            for s in batch:
                instruction = s.get('instruction','')
                user_input = s.get('input','')
                prompts.append(f"{instruction}\n\n{user_input}\n\nResponse:")
            raw_responses = self.generate_responses(prompts)
            for i, raw in enumerate(raw_responses):
                sample = batch[i]
                expected = sample.get('output','')
                extracted = self.extract_json(raw)
                # validate
                try:
                    pr = json.loads(extracted)
                    sentiment = pr.get('output','')
                    valid_sent = sentiment in ['very negative','negative','neutral','positive','very positive']
                except:
                    valid_sent = False
                has_json = ('{' in extracted and '}' in extracted and 'input' in extracted and 'output' in extracted)
                results.append({
                    'sample_id': start+i,
                    'prompt': prompts[i],
                    'response': extracted,
                    'expected_output': expected,
                    'has_valid_json': has_json,
                    'has_valid_sentiment': valid_sent
                })
                if has_json and valid_sent:
                    print(f"  ✅ Sample {start+i+1}: valid")
                elif has_json:
                    print(f"  ⚠️ Sample {start+i+1}: JSON but invalid sentiment")
                else:
                    print(f"  ❌ Sample {start+i+1}: no JSON")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='Base model name or path (Qwen2.5-7B-Instruct by default)')
    parser.add_argument('--adapter_path', default='./synthesis_model_output_amazon_qwen')
    parser.add_argument('--test_dataset',
                        default='./amazon_test_1.json',
                        help='Path to Amazon test dataset (JSON file)')
    parser.add_argument('--output_file', default='./simple_evaluation_results_amazon_qwen.json')
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    with open(args.test_dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if len(data) > args.sample_size:
        data = random.sample(data, args.sample_size)

    evaluator = SimpleEvaluatorQwen(args.base_model, args.adapter_path)
    try:
        evaluator.load_model()
    except Exception:
        print('Error while loading model:')
        traceback.print_exc()
        sys.exit(1)
    try:
        results = evaluator.evaluate_dataset(data, batch_size=args.batch_size)
    except Exception:
        print('Error during evaluation:')
        traceback.print_exc()
        sys.exit(1)

    valid_json_count = sum(1 for r in results if r['has_valid_json'])
    valid_sentiment_count = sum(1 for r in results if r.get('has_valid_sentiment', False))
    out = {
        'total_samples': len(results),
        'valid_json_count': valid_json_count,
        'valid_json_rate': valid_json_count/len(results),
        'valid_sentiment_count': valid_sentiment_count,
        'valid_sentiment_rate': valid_sentiment_count/len(results),
        'results': results
    }
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('Saved results to', args.output_file)

if __name__ == '__main__':
    main()
