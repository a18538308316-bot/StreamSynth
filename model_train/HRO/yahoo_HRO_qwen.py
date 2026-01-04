#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen reward-enhanced GRPO training script for the Yahoo dataset.

Keeps the original Yahoo reward and data processing logic, and only
switches the backbone model to Qwen2.5-7B-instruct. The base model
path can be overridden via --base-model-path.
"""

import os, sys, time, json, re, torch, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset

from scripts_yahoo.data_utils import create_optimized_dataset
from scripts_yahoo.reward_functions import initialize_reward_globals, set_training_visualizer
from scripts_yahoo.training_visualizer import initialize_visualizer
from scripts_yahoo.batch_diversity_reward import initialize_batch_diversity_calculator
from scripts_yahoo.yahoo_attribute_config import (YAHOO_ATTRPROMPT_CONFIG, load_yahoo_sample_attributes, get_yahoo_labels)

DATA_FILE = "/path/to/yahoo_train_1500.json"
MERGED_MODEL_PATH = ""
OUTPUT_DIR = "/path/to/qwen_yahoo_grpo_reward_enhanced_output"
USE_MERGED_MODEL = True
READABILITY_MODEL_PATH = "/path/to/reasoning-model"
EMBEDDING_MODEL_PATH = "/path/to/all-MiniLM-L6-v2"

ENHANCED_CONFIG = {"max_train_samples":500,"num_train_epochs":2.0,"per_device_train_batch_size":4,"gradient_accumulation_steps":4,"num_generations":4,"max_completion_length":800,"logging_steps":5,"save_steps":40,"learning_rate":3e-6,"warmup_steps":20,"max_grad_norm":1.0,"dataloader_num_workers":4}
ENHANCED_SAMPLE_REWARDS_CONFIG = {"label_consistency_weight":0.1,"attribute_compliance_weight":0.1,"generation_quality_weight":0.5}
ENHANCED_BATCH_REWARDS_CONFIG = {"batch_diversity_weight":0.3}
GENERATION_CONFIG = {"do_sample":True,"temperature":0.85,"top_p":0.92,"top_k":60,"repetition_penalty":1.12,"pad_token_id":None}

def create_enhanced_reward_functions():
    from scripts_yahoo.reward_functions import reward_generation_quality_batch
    from scripts_yahoo.batch_diversity_reward import reward_batch_diversity
    yahoo_labels = get_yahoo_labels()
    canonical_labels = {re.sub(r"[^a-z0-9]+"," ", label.lower()).strip(): label for label in yahoo_labels}
    label_keywords = {
        'society & culture': ['culture','society','tradition','community','cultural','custom','festival'],
        'science & mathematics': ['science','scientific','math','physics','chemistry','biology','research'],
        'health': ['health','medical','doctor','medicine','treatment','symptom','wellness'],
        'education & reference': ['education','school','learning','study','curriculum','student','teacher'],
        'computers & internet': ['computer','software','internet','technology','digital','online','cyber'],
        'sports': ['sport','athlete','team','league','game','match','season'],
        'business & finance': ['business','finance','market','investment','economy','stock','budget'],
        'entertainment & music': ['entertainment','music','movie','film','show','artist','album'],
        'family & relationships': ['family','relationship','parent','child','marriage','dating','partner'],
        'politics & government': ['politics','government','policy','election','law','political','governance'],
    }
    def _extract_first_json_object(text):
        dec=json.JSONDecoder(); i=0
        while i < len(text):
            b=text.find('{', i)
            if b==-1: break
            try:
                obj,end=dec.raw_decode(text,b)
                if isinstance(obj,dict): return obj
            except json.JSONDecodeError: pass
            i=b+1
        return None
    def _clean_qa_text(inp):
        if not isinstance(inp,str): return ""
        c=inp.strip();
        if c.lower().startswith('text:'): c=c[5:].strip()
        return c
    def _normalize_label(label):
        if not isinstance(label,str): return ""
        return re.sub(r"[^a-z0-9]+"," ", label.lower()).strip()
    def _label_tokens(label):
        n=_normalize_label(label); return set(n.split()) if n else set()
    def _infer_label_from_text(text):
        tl=text.lower(); best=None; score=0
        for lab, kws in label_keywords.items():
            s=sum(1 for kw in kws if kw in tl)
            if s>score: best,score=lab,s
        if score==0: return ""
        nk=_normalize_label(best)
        if nk in canonical_labels: return canonical_labels[nk]
        return best
    def _extract_target_label(prompt):
        if not isinstance(prompt,str): return 'Society & Culture'
        pats=[r'Target topic label\s*\(.*?\):\s*([\w &/-]+)', r'Target topic label\s*:\s*([\w &/-]+)', r'Target label\s*:\s*([\w &/-]+)']
        for p in pats:
            m=re.search(p,prompt,flags=re.IGNORECASE)
            if m:
                cand=m.group(1).strip(); norm=_normalize_label(cand)
                if norm in canonical_labels: return canonical_labels[norm]
                if norm:
                    for norm_label, original in canonical_labels.items():
                        if _label_tokens(norm_label) <= _label_tokens(cand): return original
                return cand
        pl=prompt.lower()
        for lab in yahoo_labels:
            if lab.lower() in pl: return lab
        return 'Society & Culture'
    def _extract_attributes(prompt):
        attrs=load_yahoo_sample_attributes(prompt or ""); return {k:v for k,v in attrs.items() if isinstance(v,str) and v}
    def _score_label_match(target_label, candidate_label, qa_text):
        target_tokens=_label_tokens(target_label) or {'society','culture'}; candidate_tokens=_label_tokens(candidate_label)
        def _sim(tokens):
            if not tokens: return 0.0
            return len(target_tokens & tokens)/max(len(target_tokens),1)
        candidate_sim=_sim(candidate_tokens)
        if candidate_sim>=0.99: return 1.0
        if candidate_sim>=0.66: return 0.7
        if candidate_sim>=0.33: return 0.4
        inferred=_infer_label_from_text(qa_text); inferred_tokens=_label_tokens(inferred); inferred_sim=_sim(inferred_tokens)
        if inferred_sim>=0.9: return 0.55
        if inferred_sim>=0.5: return 0.3
        if inferred_sim>=0.3: return 0.15
        return 0.0
    reward_call_counter=0; current_training_step=0
    def enhanced_label_consistency(completions, **kwargs):
        nonlocal reward_call_counter,current_training_step
        reward_call_counter += 1; current_training_step = reward_call_counter // 4; kwargs['step']=current_training_step
        prompts=kwargs.get('prompts',[]); out=[]
        for i,c in enumerate(completions):
            prompt=prompts[i] if i < len(prompts) else ""; target=_extract_target_label(prompt); pj=_extract_first_json_object(c); qa=_clean_qa_text(pj.get('input')) if pj else c; candidate=pj.get('output') if pj else None
            out.append(float(np.clip(_score_label_match(target,candidate,qa),0.0,1.0)))
        return out
    def enhanced_attribute_compliance(completions, **kwargs):
        kwargs['step']=current_training_step; prompts=kwargs.get('prompts',[]); out=[]
        for i,c in enumerate(completions):
            prompt=prompts[i] if i < len(prompts) else ""; attrs=_extract_attributes(prompt); target=_extract_target_label(prompt); pj=_extract_first_json_object(c); qa=_clean_qa_text(pj.get('input')) if pj else c
            if not attrs or not qa: out.append(0.0); continue
            total=0.0; cnt=0
            for an,av in attrs.items():
                chk=YAHOO_ATTRPROMPT_CONFIG.get_attribute_check_function(an,av,label=target)
                try: sc=float(chk(qa))
                except Exception: sc=0.0
                total += max(0.0,min(sc,1.0)); cnt += 1
            avg= total/cnt if cnt else 0.0
            if avg>=0.75: rw=1.0
            elif avg>=0.55: rw=0.7
            elif avg>=0.35: rw=0.4
            elif avg>=0.15: rw=0.15
            else: rw=0.0
            out.append(float(rw))
        return out
    def enhanced_generation_quality(completions, **kwargs):
        kwargs['step']=current_training_step; base=reward_generation_quality_batch(completions, **kwargs); return base.tolist() if hasattr(base,'tolist') else [float(r) for r in base]
    def enhanced_batch_diversity(completions, **kwargs):
        kwargs['step']=current_training_step; base=reward_batch_diversity(completions, **kwargs); return base.tolist() if hasattr(base,'tolist') else [float(r) for r in base]
    enhanced_label_consistency.__name__="reward_label_consistency"; enhanced_attribute_compliance.__name__="reward_attribute_compliance"; enhanced_generation_quality.__name__="reward_generation_quality"; enhanced_batch_diversity.__name__="reward_batch_diversity"
    return [enhanced_label_consistency, enhanced_attribute_compliance, enhanced_generation_quality, enhanced_batch_diversity]

def load_and_process_data(data_file, max_samples=None):
    print(f"Loading synthesis data from {data_file}")
    with open(data_file,'r',encoding='utf-8') as f: data=json.load(f)
    if max_samples: data=data[:max_samples]
    print(f"Loaded {len(data)} samples")
    grpo_data=[]
    for item in data:
        full_prompt=item.get('instruction','')+'\n\n'+item.get('input','')
        input_text=item.get('input',''); output_text=item.get('output',''); target_label='Society & Culture'
        try:
            output_json=json.loads(output_text)
            if 'output' in output_json: target_label=output_json['output']
        except (json.JSONDecodeError, KeyError, TypeError): pass
        if target_label == 'Society & Culture':
            if 'Target topic label:' in input_text:
                target_label = input_text.split('Target topic label:')[1].split('\n')[0].strip()
            elif 'Target label:' in input_text:
                target_label = input_text.split('Target label:')[1].split('\n')[0].strip()
        if target_label == 'Society & Culture':
            for lab in get_yahoo_labels():
                if lab.lower() in input_text.lower(): target_label=lab; break
        grpo_data.append({'label':target_label,'generated_label':target_label,'qa_text':item.get('output',''),'original_input':item.get('input',''),'prompt':full_prompt})
    print(f"✅ GRPO dataset prepared with {len(grpo_data)} samples")
    return Dataset.from_list(grpo_data)

class SynthesisRewardCalculator:
    def __init__(self, readability_model_path, device='cuda'):
        self.device=device; self.readability_model_path=readability_model_path; print(f"✅ SynthesisRewardCalculator initialized (device: {device})")

def setup_reward_calculators():
    print("🔧 Initializing Yahoo reward-enhanced system (Qwen)...")
    yahoo_config=YAHOO_ATTRPROMPT_CONFIG
    reward_calculator=SynthesisRewardCalculator(READABILITY_MODEL_PATH)
    print("✅ Yahoo reward-enhanced calculators initialized")
    print(f"📋 Attributes: {list(yahoo_config.attributes.keys())}")
    print(f"📋 Labels: {yahoo_config.labels}")
    return reward_calculator, None, yahoo_config, None

def plot_enhanced_training_curves(trainer, output_dir):
    import matplotlib.pyplot as plt, pandas as pd
    try:
        log_history=trainer.state.log_history
        if not log_history: print("⚠️ No training logs found for plotting"); return
        steps=[]; total_rewards=[]; losses=[]; label_rewards=[]; attribute_rewards=[]; quality_rewards=[]; diversity_rewards=[]
        for le in log_history:
            if 'step' in le:
                steps.append(le['step']); total_rewards.append(le.get('reward',0)); losses.append(le.get('loss',0)); label_rewards.append(le.get('rewards/reward_label_consistency/mean',0)); attribute_rewards.append(le.get('rewards/reward_attribute_compliance/mean',0)); quality_rewards.append(le.get('rewards/reward_generation_quality/mean',0)); diversity_rewards.append(le.get('rewards/reward_batch_diversity/mean',0))
        if not steps: print("⚠️ No step data found for plotting"); return
        plt.style.use('default'); fig, axes = plt.subplots(2,3, figsize=(18,12)); fig.suptitle('Qwen Reward Enhanced GRPO (Yahoo)', fontsize=16, fontweight='bold')
        axes[0,0].plot(steps,total_rewards,'b-',linewidth=2,marker='o',markersize=3); axes[0,0].set_title('Total Reward'); axes[0,0].grid(True,alpha=0.3)
        axes[0,1].plot(steps,losses,'r-',linewidth=2,marker='s',markersize=3); axes[0,1].set_title('Loss'); axes[0,1].grid(True,alpha=0.3)
        axes[0,2].plot(steps,label_rewards,'g-',linewidth=2,marker='^',markersize=3); axes[0,2].set_title('Label'); axes[0,2].set_ylim(0,1.1); axes[0,2].grid(True,alpha=0.3)
        axes[1,0].plot(steps,attribute_rewards,'m-',linewidth=2,marker='d',markersize=3); axes[1,0].set_title('Attribute'); axes[1,0].set_ylim(0,1.1); axes[1,0].grid(True,alpha=0.3)
        axes[1,1].plot(steps,quality_rewards,'orange',linewidth=2,marker='*',markersize=4); axes[1,1].set_title('Quality'); axes[1,1].set_ylim(0,1.1); axes[1,1].grid(True,alpha=0.3)
        axes[1,2].plot(steps,diversity_rewards,'cyan',linewidth=2,marker='v',markersize=3); axes[1,2].set_title('Diversity'); axes[1,2].set_ylim(0,1.1); axes[1,2].grid(True,alpha=0.3)
        plt.tight_layout(); out_path=f"{output_dir}/qwen_yahoo_training_curves.png"; plt.savefig(out_path,dpi=300,bbox_inches='tight'); plt.close(); print(f"✅ Curves saved: {out_path}")
        df=pd.DataFrame({'Step':steps,'Total_Reward':total_rewards,'Loss':losses,'Label_Reward':label_rewards,'Attribute_Reward':attribute_rewards,'Quality_Reward':quality_rewards,'Diversity_Reward':diversity_rewards}); csv_path=f"{output_dir}/qwen_yahoo_training_metrics.csv"; df.to_csv(csv_path,index=False); print(f"✅ Metrics saved: {csv_path}")
    except Exception as e:
        print(f"❌ Plot error: {e}"); import traceback; traceback.print_exc()

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Qwen Reward Enhanced GRPO Training")
    # Key change: optional argument with empty default (no hard-coded path)
    parser.add_argument("--base-model-path", type=str, default="", 
                      help="Base model path (provided externally)")
    parser.add_argument("--dry-run", action="store_true", help="Initialize only, skip training")
    parser.add_argument("--override-max-train-samples", type=int, default=None)
    args, unknown = parser.parse_known_args(argv)
    return args

def main():
    args = parse_args()
    global MERGED_MODEL_PATH
    
    # Priority: command-line argument first, then injected variable as fallback
    if args.base_model_path:
        MERGED_MODEL_PATH = args.base_model_path
    # If command-line argument is empty, rely on injected MERGED_MODEL_PATH (set by pipeline)
    
    # Validate that the model path exists
    if not MERGED_MODEL_PATH or not os.path.exists(MERGED_MODEL_PATH):
        print(f"❌ Invalid model path: {MERGED_MODEL_PATH}")
        return
    
    print(f"🌟 Starting Yahoo Qwen reward-enhanced GRPO training with model path: {MERGED_MODEL_PATH}")
    print("🌟 Starting Yahoo Qwen reward-enhanced GRPO training...")
    visualizer=initialize_visualizer(OUTPUT_DIR)
    reward_calculator, novelsum_calculator, attr_loader, compliance_calculator = setup_reward_calculators()
    effective_max = args.override_max_train_samples or ENHANCED_CONFIG['max_train_samples']
    if args.override_max_train_samples: print(f"⚙️ Override max_train_samples -> {effective_max}")
    dataset, training_data_global = create_optimized_dataset(DATA_FILE, effective_max, ENHANCED_CONFIG['per_device_train_batch_size'])
    initialize_reward_globals(training_data_global, ENHANCED_CONFIG['per_device_train_batch_size'], reward_calculator, novelsum_calculator, attr_loader, compliance_calculator, optimized_sample_config=ENHANCED_SAMPLE_REWARDS_CONFIG, optimized_batch_config=ENHANCED_BATCH_REWARDS_CONFIG, embedding_model_path=EMBEDDING_MODEL_PATH)
    set_training_visualizer(visualizer)
    if USE_MERGED_MODEL and os.path.exists(MERGED_MODEL_PATH):
        print("🤖 Loading Qwen model (4bit)...")
        bnb_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
        model=AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        print(f"✅ Qwen model loaded successfully hidden_size={getattr(model.config,'hidden_size','N/A')} path={MERGED_MODEL_PATH}")
    else:
        print("❌ Model path does not exist:", MERGED_MODEL_PATH); return
    tokenizer=AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token; tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.padding_side="left"
    if hasattr(model,'generation_config'):
        gc=model.generation_config; gc.do_sample=True; gc.top_p=GENERATION_CONFIG['top_p']; gc.top_k=GENERATION_CONFIG['top_k']; gc.repetition_penalty=GENERATION_CONFIG['repetition_penalty']; gc.temperature=GENERATION_CONFIG['temperature']; gc.max_new_tokens=ENHANCED_CONFIG['max_completion_length']; gc.pad_token_id=tokenizer.eos_token_id; gc.eos_token_id=tokenizer.eos_token_id
        print("🔥 Generation configuration set")
    reward_functions=create_enhanced_reward_functions(); print("✅ Reward functions ready")
    reward_weights=[ENHANCED_SAMPLE_REWARDS_CONFIG['label_consistency_weight'], ENHANCED_SAMPLE_REWARDS_CONFIG['attribute_compliance_weight'], ENHANCED_SAMPLE_REWARDS_CONFIG['generation_quality_weight'], ENHANCED_BATCH_REWARDS_CONFIG['batch_diversity_weight']]
    print("📊 Reward weights:", reward_weights)
    lora_config=LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],lora_dropout=0.1,bias="none",task_type="CAUSAL_LM",inference_mode=False)
    grpo_config=GRPOConfig(learning_rate=ENHANCED_CONFIG['learning_rate'],num_train_epochs=ENHANCED_CONFIG['num_train_epochs'],per_device_train_batch_size=ENHANCED_CONFIG['per_device_train_batch_size'],gradient_accumulation_steps=ENHANCED_CONFIG['gradient_accumulation_steps'],logging_steps=ENHANCED_CONFIG['logging_steps'],save_steps=ENHANCED_CONFIG['save_steps'],warmup_steps=ENHANCED_CONFIG['warmup_steps'],max_grad_norm=ENHANCED_CONFIG['max_grad_norm'],dataloader_num_workers=ENHANCED_CONFIG['dataloader_num_workers'],output_dir=OUTPUT_DIR,num_generations=ENHANCED_CONFIG['num_generations'],max_completion_length=ENHANCED_CONFIG['max_completion_length'],reward_weights=reward_weights,temperature=GENERATION_CONFIG['temperature'],top_p=GENERATION_CONFIG['top_p'],top_k=GENERATION_CONFIG['top_k'],repetition_penalty=GENERATION_CONFIG['repetition_penalty'],remove_unused_columns=False,report_to=[])
    trainer=GRPOTrainer(model=model,reward_funcs=reward_functions,args=grpo_config,train_dataset=dataset,processing_class=tokenizer,peft_config=lora_config)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR,'qwen_yahoo_run_config.json'),'w',encoding='utf-8') as f:
        json.dump({"enhanced_config":ENHANCED_CONFIG,"sample_reward_weights":ENHANCED_SAMPLE_REWARDS_CONFIG,"batch_reward_weights":ENHANCED_BATCH_REWARDS_CONFIG,"generation_config":GENERATION_CONFIG,"effective_max_samples":effective_max,"base_model_path":MERGED_MODEL_PATH,"timestamp":time.time()}, f, ensure_ascii=False, indent=2)
    if args.dry_run:
        print("🚫 Dry-run: skipping training"); return
    print("🔥 Starting Qwen Yahoo GRPO training...")
    st=time.time(); trainer.train(); et=time.time(); mins=(et-st)/60
    print(f"⏱️ Training time: {mins:.2f} minutes"); print("💾 Saving model..."); trainer.save_model(); print("📊 Plotting curves..."); plot_enhanced_training_curves(trainer, OUTPUT_DIR)
    print("="*80); print("🎉 Yahoo Qwen reward-enhanced training finished!"); print(f"Samples processed: {ENHANCED_CONFIG['max_train_samples']}"); print(f"Output directory: {OUTPUT_DIR}"); print("="*80)

if __name__ == '__main__':
    main()
