#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen版本 奖励增强版GRPO训练脚本 - MNLI数据集
保留原奖励与数据处理逻辑，替换底层模型为Qwen2.5-7B-instruct；支持 --base-model-path 覆盖。
"""

import os, sys, time, json, re, torch, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset

from scripts_MNLI.data_utils import create_optimized_dataset
from scripts_MNLI.reward_functions import initialize_reward_globals, set_training_visualizer
from scripts_MNLI.training_visualizer import initialize_visualizer
from scripts_MNLI.mnli_attribute_config import MNLI_ATTRPROMPT_CONFIG, load_mnli_sample_attributes

DATA_FILE = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/MNLI/MNLI_train_1496.json"
MERGED_MODEL_PATH = ""
OUTPUT_DIR = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/synthesis_model_train/TRL-GRPO-ohter-dataset/MNLI/qwen_mnli_grpo_reward_enhanced_output"
USE_MERGED_MODEL = True
READABILITY_MODEL_PATH = "/public/home/huzhenlin2023/paper_2_LLM_Synthesis/evaluate_model_data_continual_learning/reasoning-model"
EMBEDDING_MODEL_PATH = "/public/home/huzhenlin2023/synthetic_data/all-MiniLM-L6-v2"

ENHANCED_CONFIG = {"max_train_samples":500,"num_train_epochs":2.0,"per_device_train_batch_size":4,"gradient_accumulation_steps":4,"num_generations":4,"max_completion_length":800,"logging_steps":5,"save_steps":40,"learning_rate":3e-6,"warmup_steps":20,"max_grad_norm":1.0,"dataloader_num_workers":4}
ENHANCED_SAMPLE_REWARDS_CONFIG = {"label_consistency_weight":0.1,"attribute_compliance_weight":0.1,"generation_quality_weight":0.5}
ENHANCED_BATCH_REWARDS_CONFIG = {"batch_diversity_weight":0.3}
GENERATION_CONFIG = {"do_sample":True,"temperature":0.85,"top_p":0.92,"top_k":60,"repetition_penalty":1.12,"pad_token_id":None}

def create_enhanced_reward_functions():
    from scripts_MNLI.reward_functions import reward_generation_quality_batch
    from scripts_MNLI.batch_diversity_reward import reward_batch_diversity
    mnli_labels={'entailment','contradiction','neutral'}
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
    def _clean_nli_text(inp):
        if not isinstance(inp,str): return ""; return inp.strip()
    def _split_premise_hypothesis(text):
        if not text: return "",""
        m=re.search(r'Premise:\s*(.*?)\s*Hypothesis:\s*(.*)', text, flags=re.IGNORECASE|re.DOTALL)
        if m: return m.group(1).strip(), m.group(2).strip()
        return text.strip(), ""
    def _extract_target_label(prompt):
        if not isinstance(prompt,str): return 'neutral'
        pats=[r'Target label \(must match exactly\):\s*([^\n]+)', r'Target label:\s*([^\n]+)', r'label \(must match exactly\):\s*([^\n]+)', r'label:\s*([^\n]+)']
        for p in pats:
            m=re.search(p,prompt,flags=re.IGNORECASE)
            if m:
                cand=m.group(1).strip().lower()
                if cand in mnli_labels: return cand
        return 'neutral'
    def _extract_attributes(prompt):
        attrs=load_mnli_sample_attributes(prompt or ""); return {k:v for k,v in attrs.items() if isinstance(v,(str,dict)) and v}
    def _has_negation(text):
        if not text: return False
        return bool(re.search(r"\b(no|not|never|none|cannot|can't|n't)\b", text.lower()))
    def _token_overlap(a,b):
        a_tokens=set(re.findall(r'\w+', a.lower())); b_tokens=set(re.findall(r'\w+', b.lower()));
        if not b_tokens: return 0.0
        return len(a_tokens & b_tokens)/len(b_tokens)
    def _infer_label_from_text(premise,hypothesis):
        if not premise or not hypothesis: return 'neutral'
        pl=premise.lower(); hl=hypothesis.lower()
        if hl in pl or (_token_overlap(premise,hypothesis)>0.7 and _has_negation(premise)==_has_negation(hypothesis)):
            return 'entailment'
        prem_neg=_has_negation(premise); hyp_neg=_has_negation(hypothesis)
        if _token_overlap(premise,hypothesis)>0.4 and prem_neg != hyp_neg: return 'contradiction'
        # 注意：包含单引号的字符串需要使用双引号或转义，防止语法错误
        if any(w in hl for w in ["cannot","can't","never","no "]) and not prem_neg: return 'contradiction'
        return 'neutral'
    def _score_label_alignment(target_label,json_label,premise,hypothesis):
        json_label=json_label.lower() if isinstance(json_label,str) else None
        if json_label not in mnli_labels: json_label=None
        inferred=_infer_label_from_text(premise,hypothesis)
        if inferred == target_label:
            return 1.0 if json_label == target_label else 0.8
        if json_label == target_label: return 0.45
        if inferred in mnli_labels and json_label in mnli_labels:
            if {inferred,target_label} <= {'neutral','entailment'} or {inferred,target_label} <= {'neutral','contradiction'}:
                return 0.25
        return 0.0
    reward_call_counter=0; current_training_step=0
    def enhanced_label_consistency(completions, **kwargs):
        nonlocal reward_call_counter,current_training_step
        reward_call_counter += 1; current_training_step = reward_call_counter // 4; kwargs['step']=current_training_step
        prompts=kwargs.get('prompts',[]); out=[]
        for i,c in enumerate(completions):
            prompt=prompts[i] if i < len(prompts) else ""; target=_extract_target_label(prompt); pj=_extract_first_json_object(c); json_label=pj.get('output') if pj else None; qa=_clean_nli_text(pj.get('input')) if pj else c; premise,hypothesis=_split_premise_hypothesis(qa); out.append(float(np.clip(_score_label_alignment(target,json_label,premise,hypothesis),0.0,1.0)))
        return out
    def enhanced_attribute_compliance(completions, **kwargs):
        kwargs['step']=current_training_step; prompts=kwargs.get('prompts',[]); out=[]
        for i,c in enumerate(completions):
            prompt=prompts[i] if i < len(prompts) else ""; attrs=_extract_attributes(prompt); target=_extract_target_label(prompt); pj=_extract_first_json_object(c); qa=_clean_nli_text(pj.get('input')) if pj else c; premise,hypothesis=_split_premise_hypothesis(qa)
            if not attrs or not qa: out.append(0.0); continue
            total=0.0; cnt=0
            for an,av in attrs.items():
                if an=='target_label': continue
                chk=MNLI_ATTRPROMPT_CONFIG.get_attribute_check_function(an,av,label=target)
                try:
                    if an=='length_premise': attr_in=premise
                    elif an=='length_hypothesis': attr_in=hypothesis
                    else: attr_in=qa
                    sc=float(chk(attr_in))
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
        input_text=item.get('input',''); output_text=item.get('output',''); target_label='neutral'
        try:
            output_json=json.loads(output_text)
            if 'output' in output_json: target_label=output_json['output']
        except (json.JSONDecodeError, KeyError, TypeError): pass
        if target_label=='neutral' and 'Target label (must match exactly):' in input_text:
            target_label = input_text.split('Target label (must match exactly):')[1].split('\n')[0].strip()
        if target_label not in ['entailment','contradiction','neutral']: target_label='neutral'
        grpo_data.append({'label':target_label,'generated_label':target_label,'nli_text':item.get('output',''),'original_input':item.get('input',''),'prompt':full_prompt})
    print(f"✅ GRPO数据集准备完成，共{len(grpo_data)}个样本")
    return Dataset.from_list(grpo_data)

class SynthesisRewardCalculator:
    def __init__(self, readability_model_path, device='cuda'):
        self.device=device; self.readability_model_path=readability_model_path; print(f"✅ SynthesisRewardCalculator初始化完成 (设备: {device})")

def setup_reward_calculators():
    print("🔧 初始化MNLI奖励增强系统 (Qwen)...")
    mnli_config=MNLI_ATTRPROMPT_CONFIG; reward_calculator=SynthesisRewardCalculator(READABILITY_MODEL_PATH)
    print("✅ MNLI奖励增强计算器初始化完成")
    print(f"📋 属性: {list(mnli_config.attributes.keys())}")
    print(f"📋 标签: {mnli_config.labels}")
    return reward_calculator, None, mnli_config, None

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
        plt.style.use('default'); fig, axes = plt.subplots(2,3, figsize=(18,12)); fig.suptitle('Qwen Reward Enhanced GRPO (MNLI)', fontsize=16, fontweight='bold')
        axes[0,0].plot(steps,total_rewards,'b-',linewidth=2,marker='o',markersize=3); axes[0,0].set_title('Total Reward'); axes[0,0].grid(True,alpha=0.3)
        axes[0,1].plot(steps,losses,'r-',linewidth=2,marker='s',markersize=3); axes[0,1].set_title('Loss'); axes[0,1].grid(True,alpha=0.3)
        axes[0,2].plot(steps,label_rewards,'g-',linewidth=2,marker='^',markersize=3); axes[0,2].set_title('Label'); axes[0,2].set_ylim(0,1.1); axes[0,2].grid(True,alpha=0.3)
        axes[1,0].plot(steps,attribute_rewards,'m-',linewidth=2,marker='d',markersize=3); axes[1,0].set_title('Attribute'); axes[1,0].set_ylim(0,1.1); axes[1,0].grid(True,alpha=0.3)
        axes[1,1].plot(steps,quality_rewards,'orange',linewidth=2,marker='*',markersize=4); axes[1,1].set_title('Quality'); axes[1,1].set_ylim(0,1.1); axes[1,1].grid(True,alpha=0.3)
        axes[1,2].plot(steps,diversity_rewards,'cyan',linewidth=2,marker='v',markersize=3); axes[1,2].set_title('Diversity'); axes[1,2].set_ylim(0,1.1); axes[1,2].grid(True,alpha=0.3)
        plt.tight_layout(); out_path=f"{output_dir}/qwen_mnli_training_curves.png"; plt.savefig(out_path,dpi=300,bbox_inches='tight'); plt.close(); print(f"✅ Curves saved: {out_path}")
        df=pd.DataFrame({'Step':steps,'Total_Reward':total_rewards,'Loss':losses,'Label_Reward':label_rewards,'Attribute_Reward':attribute_rewards,'Quality_Reward':quality_rewards,'Diversity_Reward':diversity_rewards}); csv_path=f"{output_dir}/qwen_mnli_training_metrics.csv"; df.to_csv(csv_path,index=False); print(f"✅ Metrics saved: {csv_path}")
    except Exception as e:
        print(f"❌ Plot error: {e}"); import traceback; traceback.print_exc()

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Qwen Reward Enhanced GRPO Training")
    # 关键修改：设为可选参数，默认值为空字符串（不硬编码路径）
    parser.add_argument("--base-model-path", type=str, default="", 
                      help="模型路径（从外部传入）")
    parser.add_argument("--dry-run", action="store_true", help="仅初始化不训练")
    parser.add_argument("--override-max-train-samples", type=int, default=None)
    args, unknown = parser.parse_known_args(argv)
    return args

def main():
    args = parse_args()
    global MERGED_MODEL_PATH
    
    # 关键逻辑：优先使用命令行参数，其次使用注入的变量（双重保障）
    if args.base_model_path:
        MERGED_MODEL_PATH = args.base_model_path
    # 若命令行参数为空，才使用注入的MERGED_MODEL_PATH（此时已由Pipeline注入）
    
    # 路径校验（必须存在，否则报错）
    if not MERGED_MODEL_PATH or not os.path.exists(MERGED_MODEL_PATH):
        print(f"❌ 模型路径无效: {MERGED_MODEL_PATH}")
        return
    
    print(f"🌟 启动MNLI Qwen奖励增强GRPO训练，使用外部传入模型路径: {MERGED_MODEL_PATH}")
    print("🌟 启动MNLI Qwen奖励增强GRPO训练...")
    visualizer=initialize_visualizer(OUTPUT_DIR)
    reward_calculator, novelsum_calculator, attr_loader, compliance_calculator = setup_reward_calculators()
    effective_max = args.override_max_train_samples or ENHANCED_CONFIG['max_train_samples']
    if args.override_max_train_samples: print(f"⚙️ 覆盖max_train_samples -> {effective_max}")
    dataset, training_data_global = create_optimized_dataset(DATA_FILE, effective_max, ENHANCED_CONFIG['per_device_train_batch_size'])
    initialize_reward_globals(training_data_global, ENHANCED_CONFIG['per_device_train_batch_size'], reward_calculator, novelsum_calculator, attr_loader, compliance_calculator, optimized_sample_config=ENHANCED_SAMPLE_REWARDS_CONFIG, optimized_batch_config=ENHANCED_BATCH_REWARDS_CONFIG, embedding_model_path=EMBEDDING_MODEL_PATH)
    set_training_visualizer(visualizer)
    if USE_MERGED_MODEL and os.path.exists(MERGED_MODEL_PATH):
        print("🤖 加载Qwen模型 (4bit)...")
        bnb_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
        model=AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        print(f"✅ Qwen模型加载成功 hidden_size={getattr(model.config,'hidden_size','N/A')} path={MERGED_MODEL_PATH}")
    else:
        print("❌ 模型不存在", MERGED_MODEL_PATH); return
    tokenizer=AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token; tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.padding_side="left"
    if hasattr(model,'generation_config'):
        gc=model.generation_config; gc.do_sample=True; gc.top_p=GENERATION_CONFIG['top_p']; gc.top_k=GENERATION_CONFIG['top_k']; gc.repetition_penalty=GENERATION_CONFIG['repetition_penalty']; gc.temperature=GENERATION_CONFIG['temperature']; gc.max_new_tokens=ENHANCED_CONFIG['max_completion_length']; gc.pad_token_id=tokenizer.eos_token_id; gc.eos_token_id=tokenizer.eos_token_id
        print("🔥 生成配置完成")
    reward_functions=create_enhanced_reward_functions(); print("✅ 奖励函数准备完成")
    reward_weights=[ENHANCED_SAMPLE_REWARDS_CONFIG['label_consistency_weight'], ENHANCED_SAMPLE_REWARDS_CONFIG['attribute_compliance_weight'], ENHANCED_SAMPLE_REWARDS_CONFIG['generation_quality_weight'], ENHANCED_BATCH_REWARDS_CONFIG['batch_diversity_weight']]
    print("📊 权重:", reward_weights)
    lora_config=LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],lora_dropout=0.1,bias="none",task_type="CAUSAL_LM",inference_mode=False)
    grpo_config=GRPOConfig(learning_rate=ENHANCED_CONFIG['learning_rate'],num_train_epochs=ENHANCED_CONFIG['num_train_epochs'],per_device_train_batch_size=ENHANCED_CONFIG['per_device_train_batch_size'],gradient_accumulation_steps=ENHANCED_CONFIG['gradient_accumulation_steps'],logging_steps=ENHANCED_CONFIG['logging_steps'],save_steps=ENHANCED_CONFIG['save_steps'],warmup_steps=ENHANCED_CONFIG['warmup_steps'],max_grad_norm=ENHANCED_CONFIG['max_grad_norm'],dataloader_num_workers=ENHANCED_CONFIG['dataloader_num_workers'],output_dir=OUTPUT_DIR,num_generations=ENHANCED_CONFIG['num_generations'],max_completion_length=ENHANCED_CONFIG['max_completion_length'],reward_weights=reward_weights,temperature=GENERATION_CONFIG['temperature'],top_p=GENERATION_CONFIG['top_p'],top_k=GENERATION_CONFIG['top_k'],repetition_penalty=GENERATION_CONFIG['repetition_penalty'],remove_unused_columns=False,report_to=[])
    trainer=GRPOTrainer(model=model,reward_funcs=reward_functions,args=grpo_config,train_dataset=dataset,processing_class=tokenizer,peft_config=lora_config)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR,'qwen_mnli_run_config.json'),'w',encoding='utf-8') as f:
        json.dump({"enhanced_config":ENHANCED_CONFIG,"sample_reward_weights":ENHANCED_SAMPLE_REWARDS_CONFIG,"batch_reward_weights":ENHANCED_BATCH_REWARDS_CONFIG,"generation_config":GENERATION_CONFIG,"effective_max_samples":effective_max,"base_model_path":MERGED_MODEL_PATH,"timestamp":time.time()}, f, ensure_ascii=False, indent=2)
    if args.dry_run:
        print("🚫 Dry-run: 跳过训练"); return
    print("🔥 开始Qwen MNLI GRPO训练...")
    st=time.time(); trainer.train(); et=time.time(); mins=(et-st)/60
    print(f"⏱️ 训练耗时: {mins:.2f}分钟"); print("💾 保存模型..."); trainer.save_model(); print("📊 绘制曲线..."); plot_enhanced_training_curves(trainer, OUTPUT_DIR)
    print("="*80); print("🎉 MNLI Qwen奖励增强训练完成！"); print(f"处理样本: {ENHANCED_CONFIG['max_train_samples']}"); print(f"输出目录: {OUTPUT_DIR}"); print("="*80)

if __name__ == '__main__':
    main()
