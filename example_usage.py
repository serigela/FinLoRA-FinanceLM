#!/usr/bin/env python3
"""
Example script showing how to prepare data and train the model
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.utils import setup_logging, ensure_directories, load_config, save_jsonl
from src.data.datasets import MultiTaskDataLoader

logger = setup_logging()

def create_sample_training_data():
    """Create sample training data for demonstration"""
    logger.info("Creating sample training data...")
    
    # Ensure data directory exists
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample FLARE data
    flare_data = [
        {
            'text': 'The company reported strong quarterly earnings with revenue growth of 15%.',
            'label': 'positive',
            'task': 'sentiment',
            'split': 'train'
        },
        {
            'text': 'Stock prices plummeted after the disappointing earnings announcement.',
            'label': 'negative',
            'task': 'sentiment',
            'split': 'train'
        },
        {
            'text': 'The market remained unchanged following the Federal Reserve meeting.',
            'label': 'neutral',
            'task': 'sentiment',
            'split': 'validation'
        },
        {
            'text': 'Tech stocks surge on positive AI developments',
            'label': 'bullish',
            'task': 'headline',
            'split': 'train'
        },
        {
            'text': 'Banking sector faces regulatory headwinds',
            'label': 'bearish',
            'task': 'headline',
            'split': 'train'
        },
        {
            'text': 'Market volatility expected to continue',
            'label': 'neutral',
            'task': 'headline',
            'split': 'validation'
        }
    ]
    
    save_jsonl(flare_data, os.path.join(data_dir, "flare_data.jsonl"))
    logger.info(f"Created {len(flare_data)} FLARE samples")
    
    # Create sample ECTSum data
    ectsum_data = [
        {
            'transcript': "Good morning, everyone. Thank you for joining our Q3 2023 earnings call. I'm pleased to report that we've had another strong quarter. Our revenue grew 12% year-over-year to $2.1 billion, driven primarily by our technology segment which saw 18% growth. Our margins improved to 35%, and we're maintaining our guidance for the full year.",
            'summary': "• Q3 revenue of $2.1B, up 12% YoY\n• Technology segment grew 18%\n• Margins improved to 35%\n• Full-year guidance maintained",
            'task': 'summarization',
            'split': 'train'
        },
        {
            'transcript': "Thank you all for joining us today. This quarter presented some challenges, particularly in our international markets due to currency headwinds. However, I'm proud of our team's execution. Domestic revenue grew 8%, and we've successfully implemented cost reduction measures that will benefit us in Q4. We remain optimistic about our long-term prospects.",
            'summary': "• International markets faced currency challenges\n• Domestic revenue up 8%\n• Cost reduction measures implemented\n• Optimistic long-term outlook",
            'task': 'summarization',
            'split': 'validation'
        }
    ]
    
    save_jsonl(ectsum_data, os.path.join(data_dir, "ectsum_data.jsonl"))
    logger.info(f"Created {len(ectsum_data)} ECTSum samples")
    
    # Create sample QA data
    qa_data = [
        {
            'context': "Apple Inc. reported revenue of $117.2 billion for Q1 2024, representing a 2% increase from the previous year. iPhone sales accounted for 52% of total revenue.",
            'question': "What was Apple's Q1 2024 revenue?",
            'answer': "$117.2 billion",
            'task': 'qa',
            'split': 'train'
        },
        {
            'context': "The Federal Reserve raised interest rates by 0.25 percentage points to combat inflation. This marks the third rate increase this year.",
            'question': "By how much did the Fed raise rates?",
            'answer': "0.25 percentage points",
            'task': 'qa',
            'split': 'validation'
        }
    ]
    
    save_jsonl(qa_data, os.path.join(data_dir, "custom_qa.jsonl"))
    logger.info(f"Created {len(qa_data)} QA samples")
    
    # Create sample stock movement data
    stock_data = [
        {
            'text': 'AAPL: Apple announces new iPhone with advanced AI features',
            'movement': 2.5,
            'symbol': 'AAPL',
            'task': 'regression',
            'split': 'train'
        },
        {
            'text': 'TSLA: Tesla recalls vehicles due to safety concerns',
            'movement': -1.8,
            'symbol': 'TSLA',
            'task': 'regression',
            'split': 'train'
        },
        {
            'text': 'MSFT: Microsoft beats earnings expectations with strong cloud growth',
            'movement': 3.2,
            'symbol': 'MSFT',
            'task': 'regression',
            'split': 'validation'
        }
    ]
    
    save_jsonl(stock_data, os.path.join(data_dir, "stock_movement.jsonl"))
    logger.info(f"Created {len(stock_data)} stock movement samples")
    
    return True

def test_data_loading():
    """Test the data loading pipeline"""
    logger.info("Testing data loading pipeline...")
    
    try:
        # Create data loader
        data_loader = MultiTaskDataLoader()
        
        # Load datasets
        train_data, val_data = data_loader.combine_datasets()
        
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        
        # Show sample data
        if train_data:
            logger.info("Sample training data:")
            for i, sample in enumerate(train_data[:3]):
                logger.info(f"  {i+1}. Task: {sample['task']}")
                logger.info(f"     Text: {sample['text'][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return False

def show_instruction_formatting():
    """Show how instruction formatting works"""
    logger.info("Demonstrating instruction formatting...")
    
    config = load_config("config/model_config.yaml")
    
    # Sentiment example
    sentiment_instruction = config['tasks']['sentiment']['instruction']
    text = "The company reported strong quarterly earnings."
    label = "positive"
    
    formatted = f"Instruction: {sentiment_instruction}\nInput: {text}\nResponse: {label}"
    
    logger.info("Sentiment Classification Example:")
    logger.info(formatted)
    logger.info("")
    
    # QA example
    qa_instruction = config['tasks']['qa']['instruction']
    context = "Apple Inc. reported revenue of $117.2 billion for Q1 2024."
    question = "What was Apple's Q1 2024 revenue?"
    answer = "$117.2 billion"
    
    input_text = f"Context: {context}\nQuestion: {question}"
    formatted_qa = f"Instruction: {qa_instruction}\nInput: {input_text}\nResponse: {answer}"
    
    logger.info("Financial Q&A Example:")
    logger.info(formatted_qa)
    logger.info("")
    
    return True

def simulate_training_steps():
    """Simulate what the training process would look like"""
    logger.info("Simulating training process...")
    
    # This is what the actual training would do:
    steps = [
        "1. Loading base model (mistralai/Mistral-7B-v0.1)",
        "2. Applying LoRA configuration (r=16, alpha=32)",
        "3. Loading and formatting training data",
        "4. Setting up multi-task trainer",
        "5. Training for 3 epochs with batch size 4",
        "6. Evaluating on validation set",
        "7. Saving trained model",
        "8. Computing final metrics"
    ]
    
    for step in steps:
        logger.info(f"   {step}")
    
    # Sample training metrics
    metrics = {
        "training_loss": 1.85,
        "validation_loss": 2.12,
        "sentiment_accuracy": 0.82,
        "headline_f1": 0.79,
        "summarization_rouge_l": 0.68,
        "qa_f1": 0.73,
        "stock_prediction_r2": 0.45
    }
    
    logger.info("Expected training results:")
    for metric, value in metrics.items():
        logger.info(f"   {metric}: {value}")
    
    return True

def main():
    """Main example function"""
    logger.info("Finance LLM LoRA Training Example")
    logger.info("=" * 50)
    
    # Ensure directories exist
    ensure_directories()
    
    # Step 1: Create sample training data
    logger.info("Step 1: Creating sample training data")
    create_sample_training_data()
    
    # Step 2: Test data loading
    logger.info("\nStep 2: Testing data loading")
    test_data_loading()
    
    # Step 3: Show instruction formatting
    logger.info("\nStep 3: Demonstrating instruction formatting")
    show_instruction_formatting()
    
    # Step 4: Simulate training
    logger.info("\nStep 4: Simulating training process")
    simulate_training_steps()
    
    logger.info("\n" + "=" * 50)
    logger.info("Example completed successfully!")
    logger.info("\nTo run actual training (requires GPU and HuggingFace access):")
    logger.info("1. Set up HuggingFace token: export HF_TOKEN=your_token")
    logger.info("2. Run: python train.py --debug --output_dir ./output/test_model")
    logger.info("3. Evaluate: python evaluate.py --model_path ./output/test_model")
    logger.info("4. Demo: python demo.py --model_path ./output/test_model")

if __name__ == "__main__":
    main()