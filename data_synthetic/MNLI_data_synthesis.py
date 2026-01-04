import os, json, random, argparse, logging, threading, math
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

MNLI_LABELS = ["entailment", "contradiction", "neutral"]
GENERAL_ATTRS = [
    'premise_domain','premise_style','hypothesis_transformation',
    'semantic_phenomenon','reasoning_type','distraction_type',
    'length_premise','length_hypothesis'
]
LABEL_SPECIFIC_ATTRS = ['label_strategy']

PROMPT_TEMPLATE = """
You are a data synthesis assistant generating ONE synthetic NLI example.
Target label: {target_label}

You must output a JSON object ONLY with fields: "input" and "output".
Structure inside input MUST follow exactly:
"Premise: <premise text>\nHypothesis: <hypothesis text>"
Output label must be exactly one of: entailment, contradiction, neutral.

Attribute intentions (DO NOT list them explicitly in the texts; integrate implicitly):
- Premise domain: {premise_domain}
- Premise style: {premise_style}
- Hypothesis transformation: {hypothesis_transformation}
- Semantic phenomenon: {semantic_phenomenon}
- Reasoning type: {reasoning_type}
- Distraction type: {distraction_type}
- Label strategy (for target label): {label_strategy}
- Length constraints: premise ~{premise_len_min}-{premise_len_max} words; hypothesis ~{hypothesis_len_min}-{hypothesis_len_max} words.

Guidelines:
1. Premise and hypothesis must be independently fluent.
2. Avoid named personal sensitive data; use generic references if needed.
3. For entailment: hypothesis must be unambiguously supported by premise (not adding unverifiable specifics).
4. For contradiction: a clear conflict must exist (negation, mutually exclusive condition, quantifier mismatch, etc.).
5. For neutral: hypothesis plausible but not entailed; it must not contradict premise.
6. Avoid trivial lexical overlaps creating accidental entailment; leverage the specified distraction_type subtly.
7. Integrate the semantic phenomenon & reasoning type. Example: numerical_reasoning -> include numbers; coreference_resolution -> pronouns with clear antecedents, etc.
8. Do NOT leak the words 'entailment', 'contradiction', 'neutral' inside premise/hypothesis.
9. No meta commentary. Only the JSON.

Reference examples (style only, never copy phrases):
-- entailment example A --
{ex_entailment_1}
-- entailment example B --
{ex_entailment_2}
-- contradiction example A --
{ex_contradiction_1}
-- contradiction example B --
{ex_contradiction_2}
-- neutral example A --
{ex_neutral_1}
-- neutral example B --
{ex_neutral_2}

Return ONLY JSON like:
{{"input": "Premise: ...\nHypothesis: ...", "output": "{target_label}"}}
"""

PROMPT_MAPPING = {"prompt": PROMPT_TEMPLATE}

#############################################
# Attribute Loading
#############################################

def load_attributes(attr_name, base_dir):
    """Load attribute values supporting both new root-level layout and legacy data/ layout.

    Priority search order:
      1. <base_dir>/prompt/gpt-3.5-turbo/
      2. <base_dir>/data/prompt/gpt-3.5-turbo/
    """
    candidate_roots = [
        os.path.join(base_dir, 'prompt', 'gpt-3.5-turbo'),
        os.path.join(base_dir, 'data', 'prompt', 'gpt-3.5-turbo')
    ]
    if attr_name in GENERAL_ATTRS:
        # search general attribute file
        for root in candidate_roots:
            file_path = os.path.join(root, f"{attr_name}.txt")
            if os.path.exists(file_path):
                try:
                    with open(file_path,'r',encoding='utf-8') as f:
                        vals = [l.strip() for l in f if l.strip()]
                    if not vals:
                        raise ValueError('empty list')
                    print(f"Loaded attribute {attr_name} from {file_path} ({len(vals)} vals)")
                    return vals
                except Exception as e:
                    print(f"Warning attr {attr_name} at {file_path}: {e}")
        print(f"Warning: attribute file for {attr_name} not found in expected locations; using placeholder")
        return ['default_value']
    else:  # label_strategy (label-specific)
        strat = {}
        found_root = None
        for root in candidate_roots:
            strat_dir = os.path.join(root, attr_name)
            if os.path.isdir(strat_dir):
                found_root = strat_dir
                break
        if not found_root:
            print(f"Warning: label_strategy directory not found; using generic_strategy placeholders")
            return {lab: ['generic_strategy'] for lab in MNLI_LABELS}
        for lab in MNLI_LABELS:
            fp = os.path.join(found_root, f"{lab}.jsonl")
            try:
                with open(fp,'r',encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]
                strat[lab] = lines if lines else ['generic_strategy']
            except Exception as e:
                print(f"Missing strategy file {fp}: {e}")
                strat[lab] = ['generic_strategy']
        print(f"Loaded label_strategy from {found_root}")
        return strat


def generate_attribute_combo(attr_dict, target_label):
    combo = {}
    for k,v in attr_dict.items():
        if isinstance(v, dict):  # label specific
            lv = v.get(target_label, [])
            combo[k] = random.choice(lv) if lv else 'generic_strategy'
        else:
            combo[k] = random.choice(v)
    return combo

#############################################
# Load Original Data for Few-shot Examples
#############################################

def load_original_data(path, max_per_label=2000):
    try:
        data = json.load(open(path,'r',encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load original MNLI: {e}")
        return []
    # Clip large for memory if needed
    random.shuffle(data)
    # we just keep some for sampling style
    return data


def group_by_label(data):
    grouped = {lab: [] for lab in MNLI_LABELS}
    for item in data:
        lab = item.get('output','').strip()
        if lab in grouped:
            grouped[lab].append(item)
    for lab, arr in grouped.items():
        print(f"Original {lab}: {len(arr)}")
    return grouped


def extract_clean(text):
    # input format: Premise: ...\nHypothesis: ...
    return text.replace('\r','').strip()


def sample_examples(grouped, shots_per_label=2):
    examples = {}
    for lab in MNLI_LABELS:
        arr = grouped.get(lab, [])
        picks = random.sample(arr, min(shots_per_label, len(arr))) if arr else []
        ex_texts = []
        for p in picks:
            inp = extract_clean(p.get('input',''))
            # shorten if extremely long
            if len(inp.split()) > 120:
                inp = ' '.join(inp.split()[:120]) + ' ...'
            ex_texts.append(inp)
        # fallback placeholder if none
        while len(ex_texts) < shots_per_label:
            ex_texts.append(f"Premise: Placeholder premise for {lab}.\nHypothesis: Placeholder hypothesis.")
        examples[lab] = ex_texts
    return examples

#############################################
# Prompt Construction
#############################################

def build_prompt(attr_dict, grouped, target_label):
    combo = generate_attribute_combo(attr_dict, target_label)
    # length ranges parse
    def parse_len(val, default_min, default_max, add):
        try:
            num = int(val)
            return num, num + add
        except:
            return default_min, default_max
    prem_min, prem_max = parse_len(combo['length_premise'], 25, 90, 40)
    hyp_min, hyp_max = parse_len(combo['length_hypothesis'], 8, 30, 12)

    examples_map = sample_examples(grouped, shots_per_label=2)

    fmt = {
        'target_label': target_label,
        'premise_domain': combo['premise_domain'],
        'premise_style': combo['premise_style'],
        'hypothesis_transformation': combo['hypothesis_transformation'],
        'semantic_phenomenon': combo['semantic_phenomenon'],
        'reasoning_type': combo['reasoning_type'],
        'distraction_type': combo['distraction_type'],
        'label_strategy': combo['label_strategy'],
        'premise_len_min': prem_min,
        'premise_len_max': prem_max,
        'hypothesis_len_min': hyp_min,
        'hypothesis_len_max': hyp_max,
        'ex_entailment_1': examples_map['entailment'][0],
        'ex_entailment_2': examples_map['entailment'][1],
        'ex_contradiction_1': examples_map['contradiction'][0],
        'ex_contradiction_2': examples_map['contradiction'][1],
        'ex_neutral_1': examples_map['neutral'][0],
        'ex_neutral_2': examples_map['neutral'][1],
    }
    prompt = PROMPT_MAPPING['prompt'].format(**fmt)
    return prompt, combo

#############################################
# Model Interface
#############################################

def create_model_interface(use_api, model_path=None, api_key=None, base_url=None, model_name=None):
    if use_api:
        if not all([api_key, base_url, model_name]):
            raise ValueError('API mode requires api_key, base_url, model_name')
        if OpenAI is None:
            raise ImportError('openai not installed')
        client = OpenAI(api_key=api_key, base_url=base_url)
        return {'type':'api','client':client,'model':model_name}
    else:
        if not model_path:
            raise ValueError('Local mode requires model_path')
        if transformers is None:
            raise ImportError('transformers missing')
        pipe = transformers.pipeline(
            'text-generation', model=model_path,
            model_kwargs={'torch_dtype': torch.bfloat16 if torch else None},
            device_map='auto'
        )
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        return {'type':'local','pipeline':pipe}


def generate(model_if, prompt, max_tokens, temperature):
    if model_if['type'] == 'api':
        try:
            resp = model_if['client'].chat.completions.create(
                model=model_if['model'],
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return resp.choices[0].message.content or ''
        except Exception as e:
            print(f"API error: {e}")
            raise
    else:
        out = model_if['pipeline'](
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
# Parsing
#############################################

def parse_output(text):
    if not text or len(text.strip()) < 10:
        return {"input": f"PARSE_FAIL_RAW: {text}", "output": random.choice(MNLI_LABELS)}
    if '</think>' in text:
        part = text.split('</think>',1)[1].strip()
        if part:
            text = part
    # JSON attempt
    if '{' in text and '}' in text:
        si = text.find('{'); ei = text.rfind('}')
        blob = text[si:ei+1]
        try:
            obj = json.loads(blob)
            inp = obj.get('input','')
            out_label = obj.get('output','').strip()
            if not inp.startswith('Premise: '):
                # attempt to locate premise/hypothesis lines
                lines = [l for l in inp.split('\n') if l.strip()]
                prem, hyp = '', ''
                for l in lines:
                    low = l.lower()
                    if low.startswith('premise:') and not prem:
                        prem = l.split(':',1)[1].strip()
                    elif low.startswith('hypothesis:') and not hyp:
                        hyp = l.split(':',1)[1].strip()
                if prem and hyp:
                    inp = f"Premise: {prem}\nHypothesis: {hyp}"
            if out_label not in MNLI_LABELS:
                out_label = random.choice(MNLI_LABELS)
            if len(inp.split()) < 4:
                raise ValueError('too short')
            return {"input": inp, "output": out_label}
        except Exception:
            pass
    # Heuristic fallback
    lines = [l for l in text.split('\n') if l.strip()]
    prem, hyp = '', ''
    for l in lines:
        low = l.lower()
        if low.startswith('premise:') and not prem:
            prem = l.split(':',1)[1].strip()
        elif low.startswith('hypothesis:') and not hyp:
            hyp = l.split(':',1)[1].strip()
    if not prem or not hyp:
        # attempt extraction by pattern
        joined = ' '.join(lines)
        if 'premise' in joined.lower() and 'hypothesis' in joined.lower():
            # crude split
            try:
                pj = joined.lower().split('premise',1)[1]
            except Exception:
                pj = joined
        merged = text[:800]
        return {"input": f"Premise: {prem or 'UNKNOWN'}\nHypothesis: {hyp or merged}", "output": random.choice(MNLI_LABELS)}
    return {"input": f"Premise: {prem}\nHypothesis: {hyp}", "output": random.choice(MNLI_LABELS)}

#############################################
# Worker
#############################################

def generate_one(args_tuple):
    (gid, target_label, model_if, attr_dict, grouped, max_tokens, temperature) = args_tuple
    try:
        prompt, combo = build_prompt(attr_dict, grouped, target_label)
        raw = generate(model_if, prompt, max_tokens, temperature)
        parsed = parse_output(raw)
        # Ensure the final ground-truth label equals the intended target label (override any parser guess)
        parsed['output'] = target_label
        parsed['target_label'] = target_label
        parsed['attributes'] = combo
        parsed['raw_generation'] = raw
        parsed['timestamp'] = datetime.now().isoformat()
        return parsed
    except Exception as e:
        return {"input": f"GEN_FAIL: {e}", "output": target_label, "error": str(e)}

#############################################
# Main
#############################################

def main():
    parser = argparse.ArgumentParser(description='MNLI prompt Synthesis')
    parser.add_argument('--use_api', action='store_true')
    parser.add_argument('--model_path', type=str, default='/path/to/llama3.1_70b')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--base_url', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_tokens', type=int, default=480)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--samples_per_label', type=int, default=700)
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load attributes
    attr_dict = {}
    for attr in GENERAL_ATTRS:
        attr_dict[attr] = load_attributes(attr, base_dir)
    attr_dict['label_strategy'] = load_attributes('label_strategy', base_dir)

    # Load original data for examples
    # dataset path resolution (root file preferred, fallback to legacy data/ folder)
    root_candidate = os.path.join(base_dir, 'MNLI_train_llama_factory.json')
    legacy_candidate = os.path.join(base_dir, 'data', 'MNLI_train_llama_factory.json')
    if os.path.exists(root_candidate):
        dataset_path = root_candidate
    elif os.path.exists(legacy_candidate):
        dataset_path = legacy_candidate
        print(f"Info: using legacy dataset path {legacy_candidate}")
    else:
        print("ERROR: MNLI_train_llama_factory.json not found in root or data/ subdirectory.")
        dataset_path = root_candidate  # will fail later in load; explicit message above
    data = load_original_data(dataset_path)
    grouped = group_by_label(data)

    # Output setup
    out_dir = os.path.join(base_dir, 'synthesis_output_prompt_mnli_api')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'mnli_synthesized_prompt.json')

    synthesized = []
    if os.path.exists(out_file):
        try:
            with open(out_file,'r',encoding='utf-8') as f:
                synthesized = json.load(f)
            print(f"Resuming with {len(synthesized)} existing samples")
        except Exception as e:
            print(f"Failed to resume: {e}")
            synthesized = []

    existing_counts = {l:0 for l in MNLI_LABELS}
    for item in synthesized:
        lab = item.get('output','')
        if lab in existing_counts:
            existing_counts[lab] += 1
    for l,c in existing_counts.items():
        print(f"Existing {l}: {c}")

    plan = []
    for l in MNLI_LABELS:
        need = max(0, args.samples_per_label - existing_counts[l])
        plan.extend([l]*need)
    random.shuffle(plan)
    print(f"Planned new generations: {len(plan)}")

    try:
        model_if = create_model_interface(
            use_api=args.use_api,
            model_path=args.model_path,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name
        )
    except Exception as e:
        print(f"Model init failed: {e}")
        return

    tasks = []
    start_idx = max(args.start_idx, len(synthesized))
    for i, lab in enumerate(plan):
        gid = start_idx + i
        tasks.append((gid, lab, model_if, attr_dict, grouped, args.max_tokens, args.temperature))

    save_lock = threading.Lock()

    def save_progress():
        with save_lock:
            with open(out_file,'w',encoding='utf-8') as f:
                json.dump(synthesized,f,ensure_ascii=False,indent=2)
            print(f"Progress saved: {len(synthesized)} samples")

    if args.use_api and tasks:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            fut_to_id = {executor.submit(generate_one, t): t[0] for t in tasks}
            for idx, fut in enumerate(as_completed(fut_to_id)):
                res = fut.result()
                synthesized.append(res)
                if (idx+1) % args.save_interval == 0:
                    save_progress()
    else:
        for idx, t in enumerate(tasks):
            res = generate_one(t)
            synthesized.append(res)
            if (idx+1) % args.save_interval == 0:
                save_progress()

    save_progress()

    final_counts = {l:0 for l in MNLI_LABELS}
    attr_coverage = {a:set() for a in GENERAL_ATTRS + LABEL_SPECIFIC_ATTRS}
    for item in synthesized:
        lab = item.get('output','')
        if lab in final_counts:
            final_counts[lab] += 1
        attrs = item.get('attributes',{})
        for k,v in attrs.items():
            if k in attr_coverage:
                attr_coverage[k].add(v)

    coverage_stats = {k: len(v) for k,v in attr_coverage.items()}

    stats = {
        'total_synthesized': len(synthesized),
        'label_distribution': final_counts,
        'attribute_coverage_counts': coverage_stats,
        'attribute_categories': list(attr_dict.keys()),
        'samples_per_label_target': args.samples_per_label,
        'completion_time': datetime.now().isoformat(),
        'model': {
            'use_api': args.use_api,
            'model_name': args.model_name if args.use_api else None,
            'model_path': args.model_path if not args.use_api else None,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature
        }
    }
    stats_file = os.path.join(out_dir, 'mnli_synthesis_stats_prompt.json')
    with open(stats_file,'w',encoding='utf-8') as f:
        json.dump(stats,f,ensure_ascii=False,indent=2)
    print(f"Stats saved: {stats_file}")
    print('MNLI synthesis complete.')

if __name__ == '__main__':
    main()
