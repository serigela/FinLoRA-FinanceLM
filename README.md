 Project Overview

This project fine-tunes a 7B open-source LLM using LoRA adapters on multiple finance-specific tasks:
	•	Sentiment & Headline Classification: From the FLARE benchmark
	•	Earnings-Call Summarization: ECTSum dataset
	•	Financial Q&A: Custom dataset from FLARE/SEC filings
	•	Stock Price Regression: Predict next-day percentage change

Models are trained efficiently, tracked with W&B, and deployed via a quick REST API or Gradio demo.

⸻

 Features
	•	Parameter-efficient LoRA fine-tuning (r=16, α=32)
	•	Merged multi-task training pipeline
	•	Comprehensive evaluation with metrics: accuracy, F1, ROUGE, BERTScore, R²/MSE
	•	Experiment tracking using Weights & Biases
	•	Demo-ready deployment with Gradio and vector-backed RAG

git clone https://github.com/your-org/FinLoRA-FinanceLM.git
cd FinLoRA-FinanceLM
conda env create -f environment.yml
conda activate finlora

python scripts/download_data.py
python scripts/preprocess.py --tasks flare sentiment summarization qa regression

Output: standardized instruction–input–response .jsonl files and HF Datasets for train/validation.

Steps:
	1.	Load base model & tokenizer (mistral-7b or llama-2-7b)
	2.	Apply LoRA adapters on keys: q_proj, v_proj, k_proj, o_proj
	3.	Multi-task Trainer using Hugging Face + PEFT

python train.py \
  --model mistral-7b \
  --lora_r 16 --lora_alpha 32 \
  --tasks flare ectsum qa regression \
  --epochs 3 --batch_size 4 \
  --log wandb

python evaluate.py \
  --model output/finlora \
  --tasks flare ectsum qa regression

Metrics reported:
	•	Sentiment: Acc, F1
	•	Summarization: ROUGE-1/2/L, BERTScore
	•	QA: EM, F1
	•	Regression: MSE, R²
