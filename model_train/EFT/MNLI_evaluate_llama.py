#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concise MNLI model evaluation script.

Focuses on natural language inference: load model, generate
inference responses, extract JSON, and save results.
"""

import os
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict

class MNLIEvaluator:
    def __init__(self, base_model_path: str, adapter_path: str):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load base model and LoRA adapter."""
        print(f"Loading base model: {self.base_model_path}")
        print(f"Loading MNLI adapter: {self.adapter_path}")

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
        
        print("MNLI model loaded successfully!")
    
    def generate_response(self, prompt: str) -> str:
        """Generate an MNLI inference response for a single sample."""
        return self.generate_responses([prompt])[0]
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate MNLI inference responses in batch."""
        responses = []

        for prompt in prompts:
            # Use the same input formatting as in MNLI training
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # MNLI uses a sequence length of 2048
            ).to(self.model.device)
            
            # Generation parameters tuned for MNLI inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # MNLI answers are relatively short
                    temperature=0.2,    # Lower temperature for more deterministic reasoning
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

    def extract_json_or_label(self, response: str) -> tuple:
        """Extract MNLI result: try JSON first, then fall back to label search."""
        # First, try to extract JSON
        json_result = self.extract_json(response)
        
        # Check if it is valid JSON
        if json_result != response and '{' in json_result and '}' in json_result:
            try:
                import json as json_module
                parsed = json_module.loads(json_result)
                if 'output' in parsed:
                    return json_result, parsed['output']
            except:
                pass
        
        # If JSON extraction fails, try to find a standalone MNLI label
        nli_labels = ['entailment', 'contradiction', 'neutral']
        response_lower = response.lower()
        
        for label in nli_labels:
            if label in response_lower:
                # Construct a simple JSON structure
                constructed_json = f'{{"input": "premise-hypothesis pair", "output": "{label}"}}'
                return constructed_json, label
        
        # If everything fails, return the raw response
        return response, ""
    
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

    def evaluate_mnli_dataset(self, test_data: List[Dict], batch_size: int = 1) -> List[Dict]:
        """Evaluate the MNLI dataset with batching and NLI label detection."""
        results = []
        total_samples = len(test_data)
        valid_labels = {'entailment', 'contradiction', 'neutral'}
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_samples = test_data[batch_start:batch_end]

            print(f"Processing MNLI batch {batch_start//batch_size + 1}: samples {batch_start+1}-{batch_end}/{total_samples}")

            # Prepare prompts for the current batch
            batch_prompts = []
            for sample in batch_samples:
                instruction = sample.get('instruction', '')
                user_input = sample.get('input', '')
                full_prompt = f"{instruction}\n\n{user_input}\n\nResponse:"
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
                
                # Extract JSON or a direct label
                extracted_json, predicted_label = self.extract_json_or_label(raw_response)
                
                # Check whether there is a valid MNLI label
                has_valid_label = predicted_label in valid_labels
                has_json = ('{' in extracted_json and '}' in extracted_json and 
                           '"input"' in extracted_json and '"output"' in extracted_json)
                
                # Save result entry
                result = {
                    "sample_id": sample_id,
                    "prompt": batch_prompts[i],
                    "response": extracted_json,
                    "predicted_label": predicted_label,
                    "expected_output": expected_output,
                    "has_valid_json": has_json,
                    "has_valid_nli_label": has_valid_label
                }
                
                results.append(result)

                status = "✅" if (has_json or has_valid_label) else "❌"
                label_info = f" (predicted: {predicted_label})" if predicted_label else ""
                print(f"  {status} Sample {sample_id+1}: Valid response{label_info}")

            return results

        def main():
            parser = argparse.ArgumentParser(description="MNLI Model Evaluation")
            parser.add_argument("--base_model", type=str,
                      default="meta-llama/Meta-Llama-3-8B-Instruct",
                      help="Base model name or path (Meta-Llama-3-8B-Instruct by default)")
    parser.add_argument("--adapter_path", type=str, 
                      default="./mnli_model_output",
                      help="Path to the MNLI adapter")
    parser.add_argument("--test_dataset", type=str,
                      default="./MNLI_test_1.json",
                      help="Path to MNLI test dataset (JSON file)")
    parser.add_argument("--output_file", type=str, 
                      default="./mnli_evaluation_results.json",
                      help="Output file path")
    parser.add_argument("--sample_size", type=int, default=100,
                      help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for model inference")
    
    args = parser.parse_args()
    
    print("=== MNLI Model Evaluation ===")
    print(f"Base model: {args.base_model}")
    print(f"MNLI adapter: {args.adapter_path}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Sample size: {args.sample_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {args.output_file}")
    print("Task: Natural Language Inference (entailment/contradiction/neutral)")

    # Load MNLI test data
    print("\nLoading MNLI test dataset...")
    with open(args.test_dataset, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} MNLI samples")

    # Optional sampling
    if len(test_data) > args.sample_size:
        import random
        test_data = random.sample(test_data, args.sample_size)
        print(f"Sampled {args.sample_size} samples for MNLI evaluation")

    # Initialize evaluator
    print("\nInitializing MNLI evaluator...")
    evaluator = MNLIEvaluator(args.base_model, args.adapter_path)
    evaluator.load_model()

    # Run evaluation
    print("\nRunning MNLI evaluation...")
    results = evaluator.evaluate_mnli_dataset(test_data, batch_size=args.batch_size)

    # Compute statistics
    valid_json_count = sum(1 for r in results if r['has_valid_json'])
    valid_label_count = sum(1 for r in results if r['has_valid_nli_label'])
    valid_response_count = sum(1 for r in results if r['has_valid_json'] or r['has_valid_nli_label'])
    
    valid_json_rate = valid_json_count / len(results)
    valid_label_rate = valid_label_count / len(results)
    valid_response_rate = valid_response_count / len(results)

    # Save results
    output_data = {
        "task": "MNLI Natural Language Inference",
        "total_samples": len(results),
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_rate,
        "valid_nli_label_count": valid_label_count,
        "valid_nli_label_rate": valid_label_rate,
        "valid_response_count": valid_response_count,
        "valid_response_rate": valid_response_rate,
        "results": results
    }
    
    print(f"\nSaving MNLI results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n=== MNLI Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Valid JSON responses: {valid_json_count} ({valid_json_rate:.2%})")
    print(f"Valid NLI labels: {valid_label_count} ({valid_label_rate:.2%})")
    print(f"Valid responses (JSON or label): {valid_response_count} ({valid_response_rate:.2%})")
    print("MNLI evaluation completed successfully!")

if __name__ == "__main__":
    main()
