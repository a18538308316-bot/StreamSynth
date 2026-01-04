import os, json, random, argparse, logging, threading
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

YAHOO_LABELS = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government"
]

GENERAL_ATTRS = [
    'question_type','answer_tone','complexity_level','user_intent',
    'evidence_expectation','style','length'
]
# label-specific: domain_subtopic
LABEL_SPECIFIC_ATTRS = ['domain_subtopic']

PROMPT_TEMPLATE = """
You are a knowledge-focused data synthesis assistant.
Your task is to generate exactly ONE synthetic QA style instance for a multi-class topic classification dataset.

Target topic label: {target_label}

You must produce a plausible triplet with the following structure:
Question: <user style natural question matching the topic>
Answer: <a coherent answer consistent with attributes>
(Assume upstream instruction is: Classify the text into one of the predefined categories.)

Attribute constraints (must be reflected realistically, not listed verbatim):
- Question type: {question_type}
- User intent: {user_intent}
- Subtopic focus (domain-specific nuance): {domain_subtopic}
- Expected evidence style: {evidence_expectation}
- Desired answer tone: {answer_tone}
- Complexity level: {complexity_level}
- Presentation style: {style}
- Target length (QA combined) between {length} and {length_max} words

Guidelines:
1. Do NOT explicitly mention these attribute names; integrate them naturally.
2. Keep internal consistency: e.g., an advanced_detailed request should yield multi-step reasoning; a basic_introductory one should be simpler.
3. Avoid sensitive personal data. Health / legal / finance answers should include a brief disclaimer if giving actionable guidance.
4. The ANSWER should not leak label list; it should just naturally answer.
5. The overall content should be representative of the target label so a classifier could learn from it (implicit topical signals).
6. DO NOT copy any example verbatim.

Reference labeled snippets (for style/label distribution only, never copy phrases):
-- Society & Culture example --
{example_society}
-- Science & Mathematics example --
{example_science}
-- Health example --
{example_health}
-- Education & Reference example --
{example_education}
-- Computers & Internet example --
{example_computers}
-- Sports example --
{example_sports}
-- Business & Finance example --
{example_business}
-- Entertainment & Music example --
{example_entertainment}
-- Family & Relationships example --
{example_family}
-- Politics & Government example --
{example_politics}

Output ONLY this exact JSON structure (no extra commentary):
{{
  "input": "Text: Question: <question text>\nAnswer: <answer text>",
  "output": "{target_label}"
}}
"""

PROMPT_MAPPING = {"prompt": PROMPT_TEMPLATE}

#############################################
# Attribute Loading
#############################################

def load_attributes(attr_name, dataset='yahoo', model='gpt-3.5-turbo', classes=None):
    base_path = f"data/{dataset}/prompt/{model}/{attr_name}"
    if attr_name in GENERAL_ATTRS:
        file_path = f"{base_path}/{attr_name}.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [l.strip() for l in f if l.strip()]
        except FileNotFoundError:
            print(f"Warning: missing {file_path}, fallback placeholder")
            return ['default_value']
    else:  # label specific domain_subtopic
        attr_dict = {}
        for label in classes:
            safe_name = label.replace(' ', '_').replace('&','_')
            # We stored with original label but replaced & with _ in file creation
            # Creation used original with &? We used e.g. 'Society_&_Culture.jsonl' so preserve & pattern
            file_name = label.replace(' ', '_').replace(' & ', '_&_') + '.jsonl'
            # handle direct mapping to created file names (already created pattern with underscores and &)
            # Simpler: rebuild the actual existing file pattern
            if label == 'Society & Culture': file_name = 'Society_&_Culture.jsonl'
            elif label == 'Science & Mathematics': file_name = 'Science_&_Mathematics.jsonl'
            elif label == 'Education & Reference': file_name = 'Education_&_Reference.jsonl'
            elif label == 'Computers & Internet': file_name = 'Computers_&_Internet.jsonl'
            elif label == 'Business & Finance': file_name = 'Business_&_Finance.jsonl'
            elif label == 'Entertainment & Music': file_name = 'Entertainment_&_Music.jsonl'
            elif label == 'Family & Relationships': file_name = 'Family_&_Relationships.jsonl'
            elif label == 'Politics & Government': file_name = 'Politics_&_Government.jsonl'
            elif label == 'Sports': file_name = 'Sports.jsonl'
            elif label == 'Health': file_name = 'Health.jsonl'
            file_path = f"{base_path}/{file_name}"
            try:
                with open(file_path,'r',encoding='utf-8') as f:
                    attr_dict[label] = [l.strip() for l in f if l.strip()]
            except FileNotFoundError:
                print(f"Warning: {file_path} not found; placeholder used")
                attr_dict[label] = ['generic_subtopic']
        return attr_dict


def generate_attribute_combination(attr_dict, target_label):
    combo = {}
    for name, values in attr_dict.items():
        if isinstance(values, dict):
            if target_label in values and values[target_label]:
                combo[name] = random.choice(values[target_label])
            else:
                # fallback: random first non-empty
                fallback_list = next(iter(values.values()))
                combo[name] = random.choice(fallback_list)
        else:
            combo[name] = random.choice(values)
    return combo

#############################################
# Data Loading & Grouping
#############################################

def load_yahoo_data():
    try:
        with open('data/yahoo_train_llama_factory.json','r',encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded Yahoo dataset: {len(data)} samples")
        return data
    except Exception as e:
        print(f"Failed to load Yahoo data: {e}")
        return []


def group_by_label(data):
    grouped = {l: [] for l in YAHOO_LABELS}
    for item in data:
        lab = item.get('output','').strip()
        if lab in grouped:
            grouped[lab].append(item)
    for l, arr in grouped.items():
        print(f"Label '{l}': {len(arr)} samples")
    return grouped


def sample_examples(grouped):
    # sample one example per label for stylistic grounding
    examples = {}
    for lab in YAHOO_LABELS:
        arr = grouped.get(lab, [])
        if arr:
            pick = random.choice(arr)
            inp = pick.get('input','')
            if inp.startswith('Text: '):
                inp = inp[6:]
            # shorten overly long
            if len(inp.split()) > 120:
                inp = ' '.join(inp.split()[:120]) + ' ...'
            examples[lab] = inp
        else:
            examples[lab] = f"Example unavailable for {lab}."
    return examples

#############################################
# Prompt Construction
#############################################

def build_prompt(attr_dict, grouped, target_label):
    examples_map = sample_examples(grouped)
    combo = generate_attribute_combination(attr_dict, target_label)
    # length handling
    length_val = combo['length']
    try:
        ln = int(length_val)
        length_min = ln
        length_max = ln + 80
    except ValueError:
        length_min = 120
        length_max = 200

    format_dict = {
        'target_label': target_label,
        'question_type': combo['question_type'],
        'user_intent': combo['user_intent'],
        'domain_subtopic': combo['domain_subtopic'],
        'evidence_expectation': combo['evidence_expectation'],
        'answer_tone': combo['answer_tone'],
        'complexity_level': combo['complexity_level'],
        'style': combo['style'],
        'length': str(length_min),
        'length_max': str(length_max),
        # examples by label
        'example_society': examples_map['Society & Culture'],
        'example_science': examples_map['Science & Mathematics'],
        'example_health': examples_map['Health'],
        'example_education': examples_map['Education & Reference'],
        'example_computers': examples_map['Computers & Internet'],
        'example_sports': examples_map['Sports'],
        'example_business': examples_map['Business & Finance'],
        'example_entertainment': examples_map['Entertainment & Music'],
        'example_family': examples_map['Family & Relationships'],
        'example_politics': examples_map['Politics & Government']
    }
    prompt = PROMPT_MAPPING['prompt'].format(**format_dict)
    return prompt, combo

#############################################
# Model Interface & Generation
#############################################

def create_model_interface(use_api, model_path=None, api_key=None, base_url=None, model_name=None):
    if use_api:
        if not all([api_key, base_url, model_name]):
            raise ValueError('API mode requires api_key, base_url, model_name')
        if OpenAI is None:
            raise ImportError('openai lib missing')
        client = OpenAI(api_key=api_key, base_url=base_url)
        return {"type":"api", "client": client, "model": model_name}
    else:
        if not model_path:
            raise ValueError('Local mode requires model_path')
        if transformers is None:
            raise ImportError('transformers missing')
        pipe = transformers.pipeline(
            'text-generation', model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16 if torch else None},
            device_map='auto'
        )
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        return {"type":"local", "pipeline": pipe}


def generate(model_interface, prompt, max_tokens, temperature):
    if model_interface['type'] == 'api':
        try:
            resp = model_interface['client'].chat.completions.create(
                model=model_interface['model'],
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            content = resp.choices[0].message.content or ''
            return content
        except Exception as e:
            print(f"API error: {e}")
            raise
    else:
        out = model_interface['pipeline'](
            prompt,
            max_new_tokens=max_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )
        if isinstance(out, list) and out:
            return out[0].get('generated_text','')
        return str(out)

#############################################
# Output Parsing
#############################################

def parse_output(text, valid_labels):
    if not text or len(text.strip()) < 10:
        return {"input": f"RAW_PARSE_FAIL: {text}", "output": random.choice(valid_labels)}
    if '</think>' in text:
        part = text.split('</think>',1)[1].strip()
        if part:
            text = part
    if '{' in text and '}' in text:
        si = text.find('{'); ei = text.rfind('}')
        blob = text[si:ei+1]
        try:
            obj = json.loads(blob)
            inp = obj.get('input','')
            out_label = obj.get('output','').strip()
            if inp.startswith('Text: '):
                inp = inp[6:]
            if out_label not in valid_labels:
                out_label = random.choice(valid_labels)
            if len(inp) > 10:
                return {"input": inp, "output": out_label}
        except Exception:
            pass
    # fallback heuristic
    lines = [l for l in text.split('\n') if l.strip()]
    question, answer, label_guess = '', '', ''
    for ln in lines:
        low = ln.lower()
        if low.startswith('question:') and not question:
            question = ln.split(':',1)[1].strip()
        elif low.startswith('answer:') and not answer:
            answer = ln.split(':',1)[1].strip()
        elif '"output"' in low or low.startswith('output:'):
            for lab in valid_labels:
                if lab.lower() in low:
                    label_guess = lab
                    break
    merged = f"Question: {question}\nAnswer: {answer}".strip()
    if len(merged) < 20:
        merged = text[:4000]
    if label_guess not in valid_labels:
        label_guess = random.choice(valid_labels)
    return {"input": f"Text: {merged}", "output": label_guess}

#############################################
# Worker
#############################################

def generate_one(args_tuple):
    (sample_id, target_label, model_interface, attr_dict, grouped, max_tokens, temperature) = args_tuple
    try:
        prompt, combo = build_prompt(attr_dict, grouped, target_label)
        raw = generate(model_interface, prompt, max_tokens, temperature)
        parsed = parse_output(raw, YAHOO_LABELS)
        parsed['attributes'] = combo
        parsed['target_label'] = target_label
        parsed['raw_generation'] = raw
        parsed['timestamp'] = datetime.now().isoformat()
        return parsed
    except Exception as e:
        return {"input": f"GENERATION_EXCEPTION: {e}", "output": target_label, "error": str(e)}

#############################################
# Main
#############################################

def main():
    parser = argparse.ArgumentParser(description='Yahoo Topic Data Synthesis (prompt)')
    parser.add_argument('--use_api', action='store_true')
    parser.add_argument('--model_path', type=str, default='/path/to/llama3.1_70b')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--base_url', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_tokens', type=int, default=640)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--samples_per_label', type=int, default=200)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=40)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load attributes
    print('Loading Yahoo attributes...')
    attr_dict = {}
    for attr in GENERAL_ATTRS + LABEL_SPECIFIC_ATTRS:
        print(f'  Loading {attr}...')
        attr_dict[attr] = load_attributes(attr_name=attr, dataset='yahoo', model='gpt-3.5-turbo', classes=YAHOO_LABELS)

    # Load data
    data = load_yahoo_data()
    if not data:
        print('No source data; abort.')
        return
    grouped = group_by_label(data)

    # Output setup
    out_dir = 'synthesis_output_prompt_yahoo_api'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'yahoo_synthesized_prompt.json')

    synthesized = []
    if os.path.exists(out_file):
        try:
            with open(out_file,'r',encoding='utf-8') as f:
                synthesized = json.load(f)
            print(f"Resuming: {len(synthesized)} samples loaded")
        except Exception as e:
            print(f"Resume load failed: {e}")

    existing_counts = {l:0 for l in YAHOO_LABELS}
    for item in synthesized:
        lab = item.get('output','')
        if lab in existing_counts:
            existing_counts[lab]+=1
    for l,c in existing_counts.items():
        print(f"Existing {l}: {c}")

    plan = []
    for l in YAHOO_LABELS:
        need = max(0, args.samples_per_label - existing_counts[l])
        plan.extend([l]*need)
    random.shuffle(plan)
    print(f"Planned new generations: {len(plan)}")

    # Model interface
    try:
        model_interface = create_model_interface(
            use_api=args.use_api,
            model_path=args.model_path,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name
        )
    except Exception as e:
        print(f"Model creation failed: {e}")
        return

    tasks = []
    start_idx = max(args.start_idx, len(synthesized))
    for i, lab in enumerate(plan):
        gid = start_idx + i
        tasks.append((gid, lab, model_interface, attr_dict, grouped, args.max_tokens, args.temperature))

    save_lock = threading.Lock()

    def save_progress():
        with save_lock:
            with open(out_file,'w',encoding='utf-8') as f:
                json.dump(synthesized, f, ensure_ascii=False, indent=2)
            print(f"Saved progress: {len(synthesized)} samples")

    if args.use_api and tasks:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(generate_one, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                res = fut.result()
                synthesized.append(res)
                if len(synthesized) % args.save_interval == 0:
                    save_progress()
    else:
        for t in tasks:
            res = generate_one(t)
            synthesized.append(res)
            if len(synthesized) % args.save_interval == 0:
                save_progress()

    save_progress()

    final_counts = {l:0 for l in YAHOO_LABELS}
    for item in synthesized:
        lab = item.get('output','')
        if lab in final_counts:
            final_counts[lab]+=1

    stats = {
        'total_synthesized': len(synthesized),
        'valid_labels': YAHOO_LABELS,
        'attribute_categories': list(attr_dict.keys()),
        'final_label_distribution': final_counts,
        'samples_per_label_target': args.samples_per_label,
        'synthesis_method': 'prompt',
        'completion_time': datetime.now().isoformat(),
        'model_config': {
            'use_api': args.use_api,
            'model_name': args.model_name if args.use_api else None,
            'model_path': args.model_path if not args.use_api else None,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature
        }
    }
    stats_file = os.path.join(out_dir, 'yahoo_synthesis_stats_prompt.json')
    with open(stats_file,'w',encoding='utf-8') as f:
        json.dump(stats,f,ensure_ascii=False,indent=2)
    print(f"Stats saved: {stats_file}")
    print('Yahoo synthesis complete.')

if __name__ == '__main__':
    main()
