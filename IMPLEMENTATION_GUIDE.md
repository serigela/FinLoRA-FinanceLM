# 📋 Finance LLM LoRA - Implementation Guide

## 🎯 Project Overview

This project implements a **Multi-Task Finance LLM with LoRA (Low-Rank Adaptation)** fine-tuning system. The implementation provides a complete pipeline for training, evaluating, and deploying a financial language model that can handle multiple tasks simultaneously.

## 🏗️ Architecture

### Core Components

1. **Model Architecture (`src/models/lora_config.py`)**
   - LoRA configuration and application
   - Base model loading with quantization
   - Parameter-efficient fine-tuning setup

2. **Data Pipeline (`src/data/datasets.py`)**
   - Multi-task dataset loading
   - Instruction template formatting
   - Custom dataset classes

3. **Training System (`train.py`)**
   - Multi-task trainer implementation
   - Custom loss computation
   - Comprehensive training pipeline

4. **Evaluation System (`evaluate.py`)**
   - Task-specific evaluation metrics
   - Comprehensive performance assessment
   - Results analysis and reporting

5. **Demo Interface (`demo.py` / `demo_simple.py`)**
   - Interactive Gradio interface
   - Multi-task testing capabilities
   - User-friendly demonstration

## 📊 Supported Tasks

### 1. Sentiment Analysis
- **Input**: Financial statements, news articles
- **Output**: positive, negative, neutral
- **Metrics**: Accuracy, F1-score

### 2. Headline Classification  
- **Input**: Financial headlines
- **Output**: bullish, bearish, neutral
- **Metrics**: Accuracy, F1-score

### 3. Earnings Summarization
- **Input**: Earnings call transcripts
- **Output**: Bullet-point summaries
- **Metrics**: ROUGE-1/2/L, BERTScore

### 4. Financial Q&A
- **Input**: Context + Question
- **Output**: Answer
- **Metrics**: Exact Match, F1-score

### 5. Stock Movement Prediction
- **Input**: News sentiment
- **Output**: Percentage change prediction
- **Metrics**: MSE, R²

## 🛠️ Implementation Details

### LoRA Configuration

```yaml
lora_config:
  r: 16                    # Rank of adaptation
  lora_alpha: 32           # Scaling parameter
  target_modules:          # Target attention layers
    - "q_proj"
    - "v_proj" 
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.1        # Dropout rate
  bias: "none"             # Bias handling
```

### Training Configuration

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 16
  learning_rate: 3e-4
  weight_decay: 0.01
  warmup_steps: 100
  fp16: true
  gradient_checkpointing: true
```

### Multi-Task Learning

The system implements several multi-task learning strategies:

1. **Balanced Sampling**: Equal samples per task
2. **Proportional Sampling**: Maintain original ratios
3. **Weighted Sampling**: Custom task weights

```yaml
multi_task:
  task_weights:
    sentiment: 1.0
    headline: 1.0
    summarization: 1.5
    qa: 1.2
    regression: 0.8
  sampling_strategy: "balanced"
```

## 🚀 Usage Guide

### 1. Quick Start

```bash
# Run the complete pipeline
python run_training.py --use_wandb --debug

# Or step by step
python test_setup.py
python scripts/download_data.py
python scripts/preprocess.py
python train.py
python evaluate.py
```

### 2. Training Options

```bash
# Basic training
python train.py --output_dir ./output/model

# Debug mode (smaller dataset)
python train.py --debug

# With W&B logging
python train.py --use_wandb

# Resume from checkpoint
python train.py --resume_from_checkpoint ./output/model/checkpoint-500
```

### 3. Evaluation

```bash
# Evaluate trained model
python evaluate.py --model_path ./output/model

# Save results to custom file
python evaluate.py --model_path ./output/model --output_file results.json
```

### 4. Demo Launch

```bash
# Launch interactive demo
python demo_simple.py

# Or with trained model
python demo.py --model_path ./output/model --port 7860
```

## 📈 Performance Optimization

### Memory Optimization

1. **Quantization**: Use 8-bit or 4-bit loading
2. **Gradient Checkpointing**: Reduce memory usage
3. **Batch Size**: Adjust based on available memory
4. **Gradient Accumulation**: Maintain effective batch size

### Training Efficiency

1. **Mixed Precision**: Enable FP16 training
2. **DataLoader Workers**: Parallel data loading
3. **Compiled Model**: Use `torch.compile()` for speed
4. **Efficient Optimizers**: AdamW with proper scheduling

## 🔧 Customization

### Adding New Tasks

1. **Define Task Configuration**:
```yaml
custom_task:
  name: "custom_classification"
  type: "classification"
  labels: ["label1", "label2", "label3"]
  instruction: "Classify the following text..."
```

2. **Implement Data Loading**:
```python
def load_custom_dataset(self) -> List[Dict[str, Any]]:
    # Your custom data loading logic
    return processed_data
```

3. **Add Evaluation Metrics**:
```python
def evaluate_custom_task(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
    # Your custom evaluation logic
    return metrics
```

### Model Customization

1. **Change Base Model**:
```yaml
base_model:
  name: "meta-llama/Llama-2-7b-hf"  # Or any compatible model
```

2. **Adjust LoRA Parameters**:
```yaml
lora_config:
  r: 32                    # Higher rank for better performance
  lora_alpha: 64           # Adjust scaling
  target_modules: [...]    # Add more modules
```

3. **Custom Training Arguments**:
```yaml
training:
  num_train_epochs: 5      # More epochs
  learning_rate: 1e-4      # Different learning rate
  batch_size: 8            # Larger batch size
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model or quantization

2. **Model Loading Issues**
   - Check HuggingFace credentials
   - Verify model availability
   - Update transformers library

3. **Training Instabilities**
   - Adjust learning rate
   - Use gradient clipping
   - Check data quality

4. **Poor Performance**
   - Increase training epochs
   - Adjust LoRA parameters
   - Improve data quality/quantity

### Debug Commands

```bash
# Test setup
python test_setup.py

# Check data loading
python -c "from src.data.datasets import *; test_data_loading()"

# Verify model loading
python -c "from src.models.lora_config import *; test_model_loading()"

# Training with debug logs
python train.py --debug --verbose
```

## 📚 Dependencies

### Core Libraries
- `torch>=2.0.0`: Deep learning framework
- `transformers>=4.36.0`: HuggingFace transformers
- `peft>=0.7.0`: Parameter-efficient fine-tuning
- `datasets>=2.16.0`: Dataset loading and processing
- `accelerate>=0.25.0`: Training acceleration

### Additional Libraries
- `bitsandbytes`: Quantization support
- `wandb`: Experiment tracking
- `gradio`: Demo interface
- `rouge-score`: Evaluation metrics
- `bert-score`: Semantic evaluation

## 🎯 Best Practices

### Data Preparation
1. **Quality over Quantity**: Ensure high-quality training data
2. **Balanced Datasets**: Maintain task balance
3. **Proper Formatting**: Use consistent instruction templates
4. **Validation Split**: Keep separate validation data

### Training
1. **Start Small**: Begin with debug mode
2. **Monitor Metrics**: Track training progress
3. **Save Checkpoints**: Regular model saving
4. **Experiment Tracking**: Use W&B for logging

### Evaluation
1. **Multiple Metrics**: Use task-appropriate metrics
2. **Validation Data**: Separate test set
3. **Error Analysis**: Analyze failure cases
4. **Comparison**: Compare with baselines

## 📊 Expected Results

### Performance Benchmarks
- **Sentiment Analysis**: 80-85% accuracy
- **Headline Classification**: 75-80% accuracy
- **Summarization**: 0.6-0.7 ROUGE-L
- **Q&A**: 70-75% F1 score
- **Stock Prediction**: 0.4-0.6 R²

### Training Metrics
- **Training Time**: 2-4 hours (with GPU)
- **Memory Usage**: 12-16GB GPU memory
- **Parameters**: ~40M trainable (0.6% of base model)
- **Model Size**: ~100MB (LoRA weights only)

## 🔮 Future Enhancements

1. **More Tasks**: Add more financial NLP tasks
2. **Better Models**: Experiment with larger base models
3. **Advanced Techniques**: Implement QLoRA, DoRA
4. **Real-time Data**: Live financial data integration
5. **API Deployment**: FastAPI service deployment
6. **Monitoring**: Production monitoring and alerts

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the logs in `./logs/`
- Run `python test_setup.py` to verify setup
- Create GitHub issues for bugs
- Join the community discussion

---

This implementation provides a solid foundation for financial LLM applications with room for customization and expansion based on specific needs.