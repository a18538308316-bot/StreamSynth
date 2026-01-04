import os
import json
import random
import argparse
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import transformers, torch
except ImportError:
    transformers = None
    torch = None

############################################################
# Attribute Loading
############################################################

def load_attributes(attr_name, dataset='amazon', model='gpt-3.5-turbo', classes=None):
    """Load attribute values for both general and sentiment-dependent attributes.

    General attributes: single .txt file.
    Sentiment-dependent attributes: one .jsonl file per sentiment label.
    """
    base_path = f"data/{dataset}/prompt/{model}/{attr_name}"
    general_attrs = ['product_category', 'feature_focus', 'usage_context', 'review_angle', 'style', 'length']
    if attr_name in general_attrs:
        file_path = f"{base_path}/{attr_name}.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, using default value placeholder")
            return ['default_value']
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
    combination = {}
    for attr_name, attr_values in attr_dict.items():
        if isinstance(attr_values, dict):
            # label specific
            if target_label in attr_values and attr_values[target_label]:
                combination[attr_name] = random.choice(attr_values[target_label])
            else:
                # fallback neutral or first
                if 'neutral' in attr_values and attr_values['neutral']:
                    combination[attr_name] = random.choice(attr_values['neutral'])
                else:
                    first_list = next(iter(attr_values.values()))
                    combination[attr_name] = random.choice(first_list)
        else:
            # general attribute list
            combination[attr_name] = random.choice(attr_values)
    return combination

############################################################
# Prompt Template (Amazon)
############################################################
AMAZON_prompt_TEMPLATE = """
You are a data synthesis assistant.
Your task is to generate exactly ONE new Amazon product review based on the given labeled examples.

Sentiment label set (must use exactly as provided):
["very negative", "negative", "neutral", "positive", "very positive"]

Label semantics (keep consistent):
- "very negative": severe dissatisfaction, strong criticism, maybe product failure.
- "negative": clear disappointment or notable issues.
- "neutral": balanced, mixed pros/cons, moderate tone.
- "positive": clear satisfaction, product works as intended.
- "very positive": enthusiastic praise, strongly recommending.

Requirements:
1. Generate ONE review only. DO NOT copy sentences from examples.
2. Preserve sentiment label EXACTLY: {target_label}.
3. Obey all attribute constraints:
   - Product category context: {product_category}
   - Primary usage context: {usage_context}
   - Review angle perspective: {review_angle}
   - Core feature focus: {feature_focus}
   - Writing style: {style}
   - Perceived price/value: {price_perception}
   - Product quality/build impression: {product_quality}
   - Packaging condition: {packaging}
   - Delivery experience: {delivery_experience}
   - Customer support experience: {customer_support}
   - Overall ownership impression: {overall_experience}
   - Length between {length} and {length_max} words
4. Keep coherent: avoid contradicting attributes (e.g., don't claim both damaged packaging and perfect packaging).
5. Tone should sound like an authentic Amazon customer review (may include light imperfections, short fragments, first-person usage).
6. Do NOT add any JSON explanation outside required output format.

Labeled example snippets (do NOT imitate wording literally):

Example (very negative):
{example_very_negative}

Example (negative #1):
{example_negative_1}

Example (negative #2):
{example_negative_2}

Example (neutral #1):
{example_neutral_1}

Example (neutral #2):
{example_neutral_2}

Example (positive #1):
{example_positive_1}

Example (positive #2):
{example_positive_2}

Example (very positive):
{example_very_positive}

Target sentiment label: {target_label}

Output ONLY this exact JSON structure (no extra text):
{{
  "input": "Text: <write your new review here>",
  "output": "{target_label}"
}}
"""

PROMPT_MAPPING = {"prompt": AMAZON_prompt_TEMPLATE}

############################################################
# Data Loading
############################################################

def load_amazon_data():
    try:
        with open('data/amazon_train_llama_factory.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded Amazon training data: {len(data)} samples")
        return data
    except Exception as e:
        print(f"Failed to load Amazon data: {e}")
        return []


def load_labels():
    return ["very negative", "negative", "neutral", "positive", "very positive"]


def group_data_by_labels(data):
    grouped = {l: [] for l in load_labels()}
    for item in data:
        label = item.get('output', '').strip()
        if label in grouped:
            grouped[label].append(item)
    for l, arr in grouped.items():
        print(f"Label '{l}': {len(arr)} samples")
    return grouped


def sample_examples_for_prompt(grouped):
    distribution = {"very negative":1, "negative":2, "neutral":2, "positive":2, "very positive":1}
    examples = []
    for label, count in distribution.items():
        if len(grouped.get(label, [])) >= count:
            for item in random.sample(grouped[label], count):
                examples.append({
                    'label': label,
                    'input': item.get('input', ''),
                    'output': item.get('output', '')
                })
        else:
            print(f"Warning: insufficient examples for {label}: have {len(grouped.get(label, []))}, need {count}")
    return examples


def get_prompt_template():
    return PROMPT_MAPPING['prompt']


def create_prompt_with_quality(attr_dict, examples, target_label):
    template = get_prompt_template()
    combo = generate_attribute_combination(attr_dict, target_label)

    # organize examples
    by_label = {}
    for ex in examples:
        l = ex['output']
        by_label.setdefault(l, []).append(ex)

    def pick(label, fallback="This product was okay overall."):
        arr = by_label.get(label, [])
        if arr:
            sample = random.choice(arr)
            text = sample['input']
            if text.startswith('Text: '):
                text = text[6:]
            return text
        return fallback

    fmt = {
        'example_very_negative': pick('very negative', "Terrible purchase, regret it."),
        'example_negative_1': pick('negative', "Not very good experience."),
        'example_negative_2': pick('negative', "Some issues made it disappointing."),
        'example_neutral_1': pick('neutral', "It works but nothing special."),
        'example_neutral_2': pick('neutral', "Mixed pros and cons."),
        'example_positive_1': pick('positive', "Good value overall."),
        'example_positive_2': pick('positive', "Happy with the purchase."),
        'example_very_positive': pick('very positive', "Fantastic item!"),
        'target_label': target_label,
        # attributes
        'product_category': combo['product_category'],
        'feature_focus': combo['feature_focus'],
        'usage_context': combo['usage_context'],
        'review_angle': combo['review_angle'],
        'style': combo['style'],
        'price_perception': combo['price_perception'],
        'product_quality': combo['product_quality'],
        'packaging': combo['packaging'],
        'delivery_experience': combo['delivery_experience'],
        'customer_support': combo['customer_support'],
        'overall_experience': combo['overall_experience']
    }

    # length handling
    length_val = combo['length']
    try:
        ln = int(length_val)
        fmt['length'] = str(ln)
        fmt['length_max'] = str(ln + 60)
    except ValueError:
        fmt['length'] = '100'
        fmt['length_max'] = '160'

    return template.format(**fmt), combo

############################################################
# Model Interface
############################################################

def create_model_interface(use_api, model_path=None, api_key=None, base_url=None, model_name=None):
    if use_api:
        if not all([api_key, base_url, model_name]):
            raise ValueError('API mode requires api_key, base_url, model_name')
        if OpenAI is None:
            raise ImportError('openai package not installed')
        client = OpenAI(api_key=api_key, base_url=base_url)
        return {"type":"api", "client": client, "model": model_name}
    else:
        if not model_path:
            raise ValueError('Local mode requires model_path')
        if transformers is None:
            raise ImportError('transformers not installed')
        pipe = transformers.pipeline(
            'text-generation', model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16 if torch else None},
            device_map='auto'
        )
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        return {"type":"local", "pipeline": pipe}


def generate_with_model(model_interface, prompt, max_tokens, temperature):
    if model_interface['type'] == 'api':
        messages = [{"role":"user", "content": prompt}]
        try:
            resp = model_interface['client'].chat.completions.create(
                model=model_interface['model'],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            content = resp.choices[0].message.content or ''
            return content
        except Exception as e:
            print(f"API error: {e}")
            raise
    else:
        output = model_interface['pipeline'](
            prompt,
            max_new_tokens=max_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )
        if isinstance(output, list) and output:
            return output[0].get('generated_text', '')
        return str(output)

############################################################
# Parsing Output
############################################################

def parse_synthesis_output(text, valid_labels):
    if not text or len(text.strip()) < 10:
        return {"input": f"RAW_OUTPUT_PARSE_FAILED: {text}", "output": "neutral"}
    # strip thinking tags if any
    if '</think>' in text:
        tail = text.split('</think>',1)[1].strip()
        if tail:
            text = tail
    # Attempt JSON extraction
    if '{' in text and '}' in text:
        sj = text.find('{'); ej = text.rfind('}')
        blob = text[sj:ej+1]
        try:
            obj = json.loads(blob)
            raw_input = obj.get('input','')
            out_label = obj.get('output','').strip()
            if raw_input.startswith('Text: '):
                raw_input = raw_input[6:]
            if out_label not in valid_labels:
                out_label = 'neutral'
            if len(raw_input.strip()) >= 10:
                return {"input": raw_input.strip(), "output": out_label}
        except Exception:
            pass
    # fallback manual
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    input_buf = []
    label_val = ''
    for ln in lines:
        if ln.lower().startswith('"output"') or ln.lower().startswith('output'):
            parts = ln.split(':',1)
            if len(parts) > 1:
                candidate = parts[1].strip().strip('",')
                if candidate in valid_labels:
                    label_val = candidate
        elif '"input"' in ln or ln.lower().startswith('input'):
            # may contain JSON style
            if ':' in ln:
                seg = ln.split(':',1)[1].strip()
                seg = seg.strip('",')
                if seg.lower().startswith('text: '):
                    seg = seg[6:]
                input_buf.append(seg)
        else:
            input_buf.append(ln)
    merged = ' '.join(input_buf)
    if len(merged) < 10:
        return {"input": f"RAW_OUTPUT_PARSE_FAILED: {text}", "output": "neutral"}
    if not label_val:
        # heuristic search label
        for lab in valid_labels:
            if f'"{lab}"' in text or lab in text.split():
                label_val = lab; break
    if label_val not in valid_labels:
        label_val = 'neutral'
    return {"input": merged[:4000], "output": label_val}

############################################################
# Single Sample Generation (thread worker)
############################################################

def generate_single_sample(args_tuple):
    (sample_id, target_label, model_interface, attr_dict, grouped_data,
     labels, max_tokens, temperature) = args_tuple
    try:
        examples = sample_examples_for_prompt(grouped_data)
        prompt, attr_combo = create_prompt_with_quality(attr_dict, examples, target_label)
        raw = generate_with_model(model_interface, prompt, max_tokens, temperature)
        parsed = parse_synthesis_output(raw, labels)
        parsed['attributes'] = attr_combo
        parsed['target_label'] = target_label
        parsed['raw_generation'] = raw
        parsed['timestamp'] = datetime.now().isoformat()
        return parsed
    except Exception as e:
        return {"input": f"GENERATION_EXCEPTION: {e}", "output": target_label, "error": str(e)}

############################################################
# Main
############################################################

def main():
    parser = argparse.ArgumentParser(description='Amazon Data Synthesis (prompt)')
    parser.add_argument('--use_api', action='store_true')
    parser.add_argument('--model_path', type=str, default='/path/to/llama3.1_70b')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--base_url', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--samples_per_label', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--max_workers', type=int, default=6)
    parser.add_argument('--save_interval', type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure working directory is project root where data/ resides
    # Assuming script placed in data_synthetic_all/data_synthetic_amazon/
    # so current directory already contains data/. If not, adjust here.

    labels = load_labels()

    attributes = [
        'product_category','feature_focus','usage_context','review_angle','style','length',
        'price_perception','product_quality','packaging','delivery_experience','customer_support','overall_experience'
    ]

    print('Loading Amazon attributes...')
    attr_dict = {}
    for attr in attributes:
        print(f'  Loading {attr}...')
        attr_dict[attr] = load_attributes(attr_name=attr, dataset='amazon', model='gpt-3.5-turbo', classes=labels)

    data = load_amazon_data()
    if not data:
        print('No source data loaded, aborting.')
        return
    grouped = group_data_by_labels(data)

    output_dir = 'synthesis_output_prompt_amazon_api'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'amazon_synthesized_prompt.json')

    synthesized = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                synthesized = json.load(f)
            print(f"Resuming from existing file: {len(synthesized)} samples")
        except Exception as e:
            print(f"Failed to load existing synthesized file: {e}")

    # count existing per label
    existing_counts = {l:0 for l in labels}
    for item in synthesized:
        lab = item.get('output','')
        if lab in existing_counts:
            existing_counts[lab]+=1
    for l,c in existing_counts.items():
        print(f"Existing synthesized {l}: {c}")

    plan = []
    for l in labels:
        needed = max(0, args.samples_per_label - existing_counts.get(l,0))
        plan.extend([l]*needed)
    random.shuffle(plan)
    print(f"Planned new generations: {len(plan)}")

    # model interface
    try:
        model_interface = create_model_interface(
            use_api=args.use_api,
            model_path=args.model_path,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name
        )
    except Exception as e:
        print(f"Model interface creation failed: {e}")
        return

    tasks = []
    start_idx = max(args.start_idx, len(synthesized))
    for i, label in enumerate(plan):
        global_id = start_idx + i
        tasks.append((global_id, label, model_interface, attr_dict, grouped, labels, args.max_tokens, args.temperature))

    save_lock = threading.Lock()

    def save_progress():
        with save_lock:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(synthesized, f, ensure_ascii=False, indent=2)
            print(f"Progress saved: {len(synthesized)} samples")

    if args.use_api and tasks:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(generate_single_sample, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                res = fut.result()
                synthesized.append(res)
                if len(synthesized) % args.save_interval == 0:
                    save_progress()
    else:
        # sequential local
        for t in tasks:
            res = generate_single_sample(t)
            synthesized.append(res)
            if len(synthesized) % args.save_interval == 0:
                save_progress()

    # final save
    save_progress()

    # stats
    final_counts = {l:0 for l in labels}
    for item in synthesized:
        lab = item.get('output','')
        if lab in final_counts:
            final_counts[lab]+=1

    stats = {
        'total_synthesized': len(synthesized),
        'valid_labels': labels,
        'attribute_categories': list(attr_dict.keys()),
        'final_label_distribution': final_counts,
        'samples_per_label_target': args.samples_per_label,
        'synthesis_method': 'prompt',
        'synthesis_completion_time': datetime.now().isoformat(),
        'model_config': {
            'use_api': args.use_api,
            'model_name': args.model_name if args.use_api else None,
            'model_path': args.model_path if not args.use_api else None,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature
        }
    }
    stats_file = os.path.join(output_dir, 'amazon_synthesis_stats_prompt.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats saved: {stats_file}")
    print('Synthesis complete.')

if __name__ == '__main__':
    main()
