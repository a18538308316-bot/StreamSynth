#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concise Yahoo evaluation script.

Focuses on core functionality: load model, generate answers,
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
        """Extract the first complete JSON object from the response.

        This implementation scans all balanced-brace substrings
        and tries to parse them as JSON, preferring objects that
        contain both "input" and "output" fields.
        """
        # More robust implementation: scan all possible balanced-brace
        # substrings and attempt to parse them as JSON
        text = response
        candidates = []

        for start_idx, ch in enumerate(text):
            if ch != '{':
                continue
            brace_count = 0
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start_idx:i+1]
                        candidates.append(candidate)
                        break

        # Try to parse candidate JSON segments, preferring objects
        # that contain both input and output
        parsed_objects = []
        for cand in candidates:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    parsed_objects.append(obj)
            except Exception:
                # Ignore candidates that fail to parse
                continue

        # Prefer objects that contain both input and output
        for obj in parsed_objects:
            if 'input' in obj and 'output' in obj:
                return json.dumps(obj, ensure_ascii=False)

        # If none contain both, return the first successfully parsed object (if any)
        if parsed_objects:
            return json.dumps(parsed_objects[0], ensure_ascii=False)

        # Fallback: if no JSON object could be parsed, return the raw response
        return response

    def _parse_label_from_text(self, text: str) -> str:
        """Try to parse a label (output field) from raw text."""
        if not text:
            return ''
        # First try to parse as JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and 'output' in obj:
                return str(obj.get('output', '')).strip()
        except Exception:
            pass

        # Then try regex to match "output": "LABEL"
        import re
        m = re.search(r'"output"\s*:\s*"([^"]+)"', text)
        if m:
            return m.group(1).strip()

        # As a last resort we could try heuristics; here we just
        # return empty string to signal that nothing was found.
        return ''

    def _normalize_label(self, label: str) -> str:
        """Normalize a label to one of the canonical topic strings."""
        canonical = ['Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference',
                     'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music',
                     'Family & Relationships', 'Politics & Government']
        if not label:
            return ''
        lab = label.strip()
        for c in canonical:
            if lab.lower() == c.lower():
                return c
        # If we cannot match exactly, fall back to returning
        # the original label with minimal normalization
        return lab
    
    def evaluate_dataset(self, test_data: List[Dict], batch_size: int = 1) -> List[Dict]:
        """Evaluate the Yahoo QA dataset with optional batching.

        Yahoo test data format:
        {
            "instruction": "...",
            "input": "Requirements: ...",
            "output": "{\"input\": \"Text: Question: ...\nAnswer: ...\", \"output\": \"Education & Reference\"}"
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
                # Yahoo format: instruction + input + a blank line
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

                # Additionally validate whether the topic label is valid for Yahoo
                try:
                    import json as json_module
                    parsed_response = json_module.loads(extracted_json)
                    topic_label = parsed_response.get('output', '')
                    valid_topic = topic_label in ['Society & Culture', 'Science & Mathematics', 'Health', 
                                                 'Education & Reference', 'Computers & Internet', 'Sports',
                                                 'Business & Finance', 'Entertainment & Music', 
                                                 'Family & Relationships', 'Politics & Government']
                except:
                    valid_topic = False
                
                # Save result entry
                result = {
                    "sample_id": sample_id,
                    "prompt": batch_prompts[i],
                    "response": extracted_json,
                    "expected_output": expected_output,
                    "has_valid_json": has_json,
                    "has_valid_topic": valid_topic
                }

                results.append(result)

                if has_json and valid_topic:
                    print(f"  ✅ Sample {sample_id+1}: Valid JSON + topic")
                elif has_json:
                    print(f"  ⚠️ Sample {sample_id+1}: Valid JSON but invalid topic")
                else:
                    print(f"  ❌ Sample {sample_id+1}: No valid JSON found")

        return results

def main():
    parser = argparse.ArgumentParser(description="Simple Model Evaluation")
    parser.add_argument("--base_model", type=str,
                      default="meta-llama/Meta-Llama-3-8B-Instruct",
                      help="Base model name or path (Meta-Llama-3-8B-Instruct by default)")
    parser.add_argument("--adapter_path", type=str, 
                      default="./synthesis_model_output_yahoo",
                      help="Path to the adapter")
    parser.add_argument("--test_dataset", type=str,
                      default="./yahoo_test_1.json",
                      help="Path to Yahoo test dataset (JSON file)")
    parser.add_argument("--output_file", type=str, 
                      default="./simple_evaluation_results_yahoo.json",
                      help="Output file path")
    parser.add_argument("--sample_size", type=int, default=100,
                      help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for model inference")
    
    args = parser.parse_args()
    
    print("=== Simple Model Evaluation ===")
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Sample size: {args.sample_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {args.output_file}")

    # Load test dataset
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
    valid_topic_count = sum(1 for r in results if r.get('has_valid_topic', False))
    valid_json_rate = valid_json_count / len(results)
    valid_topic_rate = valid_topic_count / len(results)

    # Save results
    output_data = {
        "model_type": "fine_tuned",
        "total_samples": len(results),
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_rate,
        "valid_topic_count": valid_topic_count,
        "valid_topic_rate": valid_topic_rate,
        "results": results
    }
    
    print(f"\nSaving results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n=== Yahoo QA Topic Classification Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Valid JSON responses: {valid_json_count}")
    print(f"Valid JSON rate: {valid_json_rate:.2%}")
    print(f"Valid topic responses: {valid_topic_count}")
    print(f"Valid topic rate: {valid_topic_rate:.2%}")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()

    # CLI entry point for postprocessing existing evaluation result files
    # Example: python simple_evaluate.py --postprocess_results path/to/results.json
    import argparse
    post_parser = argparse.ArgumentParser(add_help=False)
    post_parser.add_argument("--postprocess_results", type=str, default=None,
                             help="Path to an existing evaluation results JSON to postprocess and fix")
    post_args, _ = post_parser.parse_known_args()
    if post_args.postprocess_results:
        def postprocess_existing_results(results_path: str):
            print(f"🔧 Postprocessing results file: {results_path}")
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = data.get('results', []) if isinstance(data, dict) else data
            fixed = []
            fixed_count = 0
            filled_label_count = 0

            for entry in results:
                resp = entry.get('response', '')
                # 1) If the response itself is a JSON string (nested), try to parse and extract it
                extracted = None
                try:
                    # Try to parse directly
                    parsed = json.loads(resp)
                    if isinstance(parsed, dict):
                        extracted = parsed
                except Exception:
                    # Fallback to existing extract_json logic
                    extracted_str = SimpleEvaluator('', '').extract_json(resp)
                    try:
                        parsed = json.loads(extracted_str)
                        if isinstance(parsed, dict):
                            extracted = parsed
                    except Exception:
                        extracted = None

                # 2) If there is no output field, try to recover it from response or prompt
                if extracted is None:
                    # Could not parse as JSON, keep the original response
                    entry['fixed_response'] = resp
                else:
                    # Use the parsed object as the fixed response
                    entry['fixed_response'] = json.dumps(extracted, ensure_ascii=False)
                    fixed_count += 1

                    # If output is missing or empty, try to fill it from the parsed object or original prompt
                    out_label = extracted.get('output', '')
                    if not out_label:
                        # Try to parse the label from the raw response text
                        candidate = SimpleEvaluator('', '')._parse_label_from_text(resp)
                        if candidate:
                            out_label = SimpleEvaluator('', '')._normalize_label(candidate)
                    if not out_label:
                        # Finally, try to find the target label from the prompt (when 'Target topic label' is present)
                        prompt = entry.get('prompt', '')
                        # Use a simple match of labels mentioned in the prompt to fill the label
                        # Here we simply search for any canonical label that appears in the prompt text
                        for c in ['Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference',
                                  'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music',
                                  'Family & Relationships', 'Politics & Government']:
                            if c in prompt:
                                out_label = c
                                break

                    if out_label:
                        # Write the output back into the fixed_response JSON (ensure it exists)
                        try:
                            obj = json.loads(entry['fixed_response'])
                            obj['output'] = out_label
                            entry['fixed_response'] = json.dumps(obj, ensure_ascii=False)
                            entry['has_valid_json'] = True
                            entry['has_valid_topic'] = True
                            filled_label_count += 1
                        except Exception:
                            pass

                fixed.append(entry)

            # Write out the fixed file
            out_path = results_path.replace('.json', '') + '_fixed.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                if isinstance(data, dict):
                    data['results'] = fixed
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(fixed, f, indent=2, ensure_ascii=False)

            print(f"✅ Postprocess complete. Total: {len(results)}, parsed_json_fixed: {fixed_count}, labels_filled: {filled_label_count}")
            print(f"Fixed file written to: {out_path}")

        postprocess_existing_results(post_args.postprocess_results)
