<h1 align="center"> RE </h1>
<h3 align="center"> Exploration via Reasoning Estimator </h3>




  
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/HuzhouNLP/Exploration-via-Reasoning-Estimator) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/HuzhouNLP/Exploration-via-Reasoning-Estimator?color=green) 

## Table of Contents

- 🌻[Acknowledgement](#acknowledgement)
- 🌟[Overview](#overview)
- 🔧[Installation](#installation)
- 📉[Model Training](#model-training)
- 🧐[Evaluation](#evaluation)
- 🚩[Citation](#citation)

---



## 🌻Acknowledgement

Our code for the training module and the inference module is implemented based on [TRL](https://github.com/huggingface/trl). Thanks for their great contributions! 


![alt text](framework_v2_01.png)

## 🌟Overview

Large language models (LLMs) have been widely adopted for synthetic data generation, significantly reducing annotation costs. However, most existing studies treat synthesis as a set of isolated tasks and overlook a more fundamental question: whether a model can learn to synthesize by accumulating experience from past tasks and transferring it to future ones.
In this work, we introduce StreamSynth, a new setting in which synthesis tasks arrive sequentially and experience from historical tasks provides informative signals for future synthesis. Under this setting, we propose SynLearner, a unified framework for synthesis learning over a task stream that integrates Diversity-Aware Initialization, Efficient Fine-Tuning, and Hierarchical Reward Optimization, which together encourage the model to balance quality and diversity over time.
Extensive experiments across multiple benchmarks show that SynLearner effectively leverages experience from earlier tasks to improve synthesis performance on later ones, exhibiting consistent cross-task transferability. These results validate the feasibility of StreamSynth and highlight synthesis as an experience-driven process that benefits from task streams.


## 🔧Installation

```bash
pip install -r requirements.txt
```

We build on [TRL](https://github.com/huggingface/trl), which can be installed via:

```bash
pip install trl
```

## 📉Model Training

We provide unified entry scripts for three main stages: diversity-aware initialization (DAI), efficient fine-tuning (EFT), and hierarchical reward optimization (HRO).

### 1. Diversity-aware initialization

Script: `data_synthetic/run_synthesis.py`

```bash
# Yelp, local LLaMA model
python data_synthetic/run_synthesis.py \
  --dataset yelp \
  --script_args "--model_path /path/to/llama3-8b --samples_per_label"

# Amazon, via OpenAI-compatible API
python data_synthetic/run_synthesis.py \
  --dataset amazon \
  --script_args "--use_api --api_key sk-xxx --base_url https://api.your-endpoint.com --model_name your-model"
```

### 2. Efficient Fine-Tuning (EFT)

Script: `model_train/run_eft.py`

```bash
# Amazon sentiment, LLaMA, SFT+LoRA training
python model_train/run_eft.py \
  --task amazon \
  --model_family llama \
  --mode train

# Yahoo topic classification, Qwen, evaluation with custom test set
python model_train/run_eft.py \
  --task yahoo \
  --model_family qwen \
  --mode eval \
  --script_args "--test_dataset ./yahoo_test_custom.json --sample_size"
```

### 3. Hierarchical Reward Optimization (HRO)

Script: `model_train/run_hro.py`

```bash
# Amazon, LLaMA, reward-enhanced GRPO
python model_train/run_hro.py \
  --task amazon \
  --model_family llama \
  --script_args "--enable-dynamic-reward"

# Yelp, Qwen, specifying merged model path
python model_train/run_hro.py \
  --task yelp \
  --model_family qwen \
  --script_args "--base-model-path /path/to/qwen_merged_model"
```

## 🧐Evaluation

EFT evaluation is handled via the same unified entry script as training:

```bash
# Yahoo topic classification, Qwen, EFT evaluation example
python model_train/run_eft.py \
  --task yahoo \
  --model_family qwen \
  --mode eval \
  --script_args "--test_dataset ./yahoo_test_custom.json --sample_size 200"
```

For more fine-grained control (e.g., per-dataset evaluation or custom metrics), you can also directly call the dataset-specific evaluation scripts under `model_train/EFT`.

## 🚩Citation

Please cite this repository if you find it useful in your work. Thanks!

```bibtex

```



## 🎉Contributors



We will offer long-term maintenance to fix bugs and solve issues. So if you have any problems, please put issues to us.
