# 🏦 Finance LLM with LoRA Fine-tuning

This project implements a comprehensive **Multi-Task Finance LLM** using **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning. The model is trained on multiple financial tasks including sentiment analysis, headline classification, earnings summarization, financial Q&A, and stock movement prediction.

## 🎯 Features

- **Parameter-efficient LoRA fine-tuning** (r=16, α=32)
- **Multi-task training pipeline** for finance-specific tasks
- **Comprehensive evaluation** with task-specific metrics
- **Gradio demo interface** for interactive testing
- **Experiment tracking** with Weights & Biases
- **Quantization support** for memory efficiency
- **Easy-to-use training pipeline**

## 📊 Supported Tasks

1. **Sentiment Analysis**: Classify financial statements as positive, negative, or neutral
2. **Headline Classification**: Categorize financial headlines as bullish, bearish, or neutral
3. **Earnings Summarization**: Generate bullet-point summaries from earnings call transcripts
4. **Financial Q&A**: Answer questions based on financial contexts
5. **Stock Movement Prediction**: Predict percentage price changes based on news sentiment

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/finance-llm-lora.git
cd finance-llm-lora

# Create conda environment
conda env create -f environment.yml
conda activate finlora

# Or install with pip
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with your API keys:

```bash
# Hugging Face
HF_TOKEN=your_huggingface_token_here

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=finance-llm-lora

# Financial Data APIs (optional)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

## 🚀 Quick Start

### Option 1: Complete Training Pipeline

Run the complete training pipeline with one command:

```bash
python run_training.py --use_wandb --debug
```

This will:
1. Test the setup
2. Download and preprocess data
3. Train the model
4. Evaluate the model
5. Launch the demo

### Option 2: Step-by-Step

#### 1. Test Setup
```bash
python test_setup.py
```

#### 2. Download Data
```bash
cd scripts
python download_data.py --datasets all
```

#### 3. Preprocess Data
```bash
cd scripts
python preprocess.py --tasks flare ectsum qa regression
```

#### 4. Train Model
```bash
python train.py \
  --output_dir ./output/finlora \
  --use_wandb \
  --debug
```

#### 5. Evaluate Model
```bash
python evaluate.py \
  --model_path ./output/finlora \
  --output_file evaluation_results.json
```

#### 6. Launch Demo
```bash
python demo.py \
  --model_path ./output/finlora \
  --port 7860
```

## 📁 Project Structure

```
finance-llm-lora/
├── config/
│   ├── model_config.yaml      # Model configuration
│   └── training_config.yaml   # Training parameters
├── scripts/
│   ├── download_data.py       # Data downloading
│   ├── preprocess.py          # Data preprocessing
│   └── utils.py               # Utility functions
├── src/
│   ├── models/
│   │   └── lora_config.py     # LoRA model setup
│   └── data/
│       └── datasets.py        # Dataset classes
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── demo.py                    # Gradio demo
├── test_setup.py              # Setup testing
├── run_training.py            # Complete pipeline
└── README.md                  # This file
```

## ⚙️ Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
base_model:
  name: "mistralai/Mistral-7B-v0.1"
  quantization:
    load_in_8bit: true

lora_config:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  lora_dropout: 0.1
```

### Training Configuration (`config/training_config.yaml`)

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 16
  learning_rate: 3e-4
  fp16: true
```

## 📊 Evaluation Metrics

The model is evaluated using task-specific metrics:

- **Sentiment & Headlines**: Accuracy, F1-score
- **Summarization**: ROUGE-1/2/L, BERTScore
- **Q&A**: Exact Match, F1-score
- **Stock Movement**: MSE, R²

## 🖥️ Demo Interface

The Gradio demo provides an interactive interface for testing all tasks:

![Demo Interface](demo_screenshot.png)

Access the demo at: `http://localhost:7860`

## 📈 Training Results

Example training results:

```
Model Info:
- Base Model: mistralai/Mistral-7B-v0.1
- LoRA r: 16, alpha: 32
- Total Parameters: 7.24B
- Trainable Parameters: 41.9M (0.58%)

Evaluation Results:
- Sentiment Accuracy: 0.85
- Headline F1: 0.82
- Summarization ROUGE-L: 0.75
- Q&A F1: 0.78
- Stock Movement R²: 0.65
```

## 🔧 Advanced Usage

### Custom Datasets

Add your own datasets by implementing the data loading functions in `src/data/datasets.py`:

```python
def load_custom_dataset(self) -> List[Dict[str, Any]]:
    # Your custom data loading logic
    pass
```

### Model Customization

Modify the LoRA configuration in `config/model_config.yaml`:

```yaml
lora_config:
  r: 32                    # Increase for better performance
  lora_alpha: 64           # Adjust scaling
  target_modules: [...]    # Add more modules
```

### Multi-GPU Training

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `per_device_train_batch_size: 2`
   - Enable gradient checkpointing: `gradient_checkpointing: true`
   - Use 4-bit quantization: `load_in_4bit: true`

2. **Model Loading Issues**
   - Check Hugging Face token: `HF_TOKEN=your_token`
   - Verify model access permissions
   - Update transformers library: `pip install -U transformers`

3. **Data Loading Errors**
   - Run data download separately: `python scripts/download_data.py`
   - Check internet connection
   - Verify dataset availability

### Performance Tips

- Use SSD storage for faster data loading
- Increase `dataloader_num_workers` for faster preprocessing
- Enable mixed precision training with `fp16: true`
- Use gradient accumulation for larger effective batch sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Microsoft](https://github.com/microsoft/LoRA) for the LoRA implementation
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
- Financial dataset providers: FLARE, ECTSum

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Join our [Discord community](https://discord.gg/finance-llm)
- Email: support@finance-llm.com

---

**Happy Fine-tuning!** 🚀