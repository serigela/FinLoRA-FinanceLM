#!/usr/bin/env python3
"""
Test script to verify the setup and basic functionality
"""

import os
import sys
import torch
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from scripts.utils import setup_logging, load_config, ensure_directories
from src.models.lora_config import FinanceLLMLoRA
from src.data.datasets import MultiTaskDataLoader

logger = setup_logging()

def test_config_loading():
    """Test configuration loading"""
    logger.info("Testing configuration loading...")
    
    try:
        model_config = load_config("config/model_config.yaml")
        training_config = load_config("config/training_config.yaml")
        
        logger.info("✓ Configuration files loaded successfully")
        logger.info(f"Base model: {model_config['base_model']['name']}")
        logger.info(f"LoRA r: {model_config['lora_config']['r']}")
        logger.info(f"Training epochs: {training_config['training']['num_train_epochs']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False

def test_directories():
    """Test directory creation"""
    logger.info("Testing directory creation...")
    
    try:
        ensure_directories()
        
        required_dirs = [
            "./models", "./data", "./output", "./logs", "./checkpoints"
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                logger.info(f"✓ Directory exists: {dir_path}")
            else:
                logger.warning(f"⚠ Directory missing: {dir_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Directory creation failed: {e}")
        return False

def test_model_loading():
    """Test model and tokenizer loading"""
    logger.info("Testing model loading...")
    
    try:
        # Create LoRA model instance
        lora_model = FinanceLLMLoRA()
        
        # Load tokenizer first (faster)
        tokenizer = lora_model.load_tokenizer()
        logger.info("✓ Tokenizer loaded successfully")
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")
        
        # Test tokenization
        test_text = "The stock market showed positive sentiment today."
        tokens = tokenizer(test_text, return_tensors="pt")
        logger.info(f"✓ Tokenization test passed - Input shape: {tokens['input_ids'].shape}")
        
        # Test model loading (this might take a while)
        logger.info("Loading base model (this may take a few minutes)...")
        base_model = lora_model.load_base_model()
        logger.info("✓ Base model loaded successfully")
        
        # Apply LoRA
        logger.info("Applying LoRA adapters...")
        peft_model = lora_model.apply_lora(base_model)
        logger.info("✓ LoRA adapters applied successfully")
        
        # Print model info
        model_info = lora_model.get_model_info()
        logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    logger.info("Testing data loading...")
    
    try:
        # Create data loader
        data_loader = MultiTaskDataLoader()
        
        # Test individual dataset loading
        logger.info("Testing FLARE data loading...")
        flare_data = data_loader.load_flare_dataset()
        logger.info(f"✓ FLARE data: {len(flare_data)} samples")
        
        logger.info("Testing ECTSum data loading...")
        ectsum_data = data_loader.load_ectsum_dataset()
        logger.info(f"✓ ECTSum data: {len(ectsum_data)} samples")
        
        logger.info("Testing QA data loading...")
        qa_data = data_loader.load_custom_qa_dataset()
        logger.info(f"✓ QA data: {len(qa_data)} samples")
        
        logger.info("Testing stock data loading...")
        stock_data = data_loader.load_stock_movement_dataset()
        logger.info(f"✓ Stock data: {len(stock_data)} samples")
        
        # Test dataset combination
        train_data, val_data = data_loader.combine_datasets()
        logger.info(f"✓ Combined datasets: {len(train_data)} train, {len(val_data)} validation")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    logger.info("Testing CUDA availability...")
    
    try:
        if torch.cuda.is_available():
            logger.info("✓ CUDA is available")
            logger.info(f"CUDA devices: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name()}")
            
            # Test GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
            
        else:
            logger.warning("⚠ CUDA not available - using CPU")
            logger.info("CPU count: {}".format(os.cpu_count()))
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CUDA test failed: {e}")
        return False

def test_dependencies():
    """Test important dependencies"""
    logger.info("Testing dependencies...")
    
    dependencies = [
        "torch", "transformers", "peft", "datasets", "accelerate", 
        "bitsandbytes", "pandas", "numpy", "scikit-learn"
    ]
    
    all_good = True
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✓ {dep}: {version}")
        except ImportError:
            logger.error(f"✗ {dep}: not installed")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    logger.info("Starting Finance LLM setup tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_config_loading),
        ("Directories", test_directories),
        ("CUDA", test_cuda_availability),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:<20}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The setup is ready for training.")
    else:
        logger.warning("⚠ Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)