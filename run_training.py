#!/usr/bin/env python3
"""
Complete training pipeline for Finance LLM LoRA
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from scripts.utils import setup_logging, ensure_directories

logger = setup_logging()

def run_command(command, description):
    """Run a command and log the result"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"✗ {description} failed")
            logger.error(f"Error: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ {description} failed with exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete Finance LLM LoRA training pipeline")
    parser.add_argument("--skip_test", action="store_true", help="Skip initial setup test")
    parser.add_argument("--skip_data", action="store_true", help="Skip data download")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with smaller dataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--output_dir", default="./output/finlora", help="Output directory for model")
    
    args = parser.parse_args()
    
    logger.info("Starting Finance LLM LoRA training pipeline...")
    logger.info("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Step 1: Test setup
    if not args.skip_test:
        logger.info("Step 1: Testing setup...")
        if not run_command("python test_setup.py", "Setup test"):
            logger.error("Setup test failed. Please fix the issues before proceeding.")
            return False
    
    # Step 2: Download data
    if not args.skip_data:
        logger.info("Step 2: Downloading data...")
        if not run_command("cd scripts && python download_data.py --datasets all", "Data download"):
            logger.error("Data download failed.")
            return False
    
    # Step 3: Preprocess data
    if not args.skip_preprocess:
        logger.info("Step 3: Preprocessing data...")
        if not run_command("cd scripts && python preprocess.py --tasks flare ectsum qa regression", "Data preprocessing"):
            logger.error("Data preprocessing failed.")
            return False
    
    # Step 4: Train model
    logger.info("Step 4: Training model...")
    
    train_command = f"python train.py --output_dir {args.output_dir}"
    
    if args.debug:
        train_command += " --debug"
    
    if args.use_wandb:
        train_command += " --use_wandb"
    
    if not run_command(train_command, "Model training"):
        logger.error("Model training failed.")
        return False
    
    # Step 5: Evaluate model
    logger.info("Step 5: Evaluating model...")
    
    eval_command = f"python evaluate.py --model_path {args.output_dir} --output_file {args.output_dir}/evaluation_results.json"
    
    if not run_command(eval_command, "Model evaluation"):
        logger.error("Model evaluation failed.")
        return False
    
    # Step 6: Launch demo (optional)
    logger.info("Step 6: Training pipeline completed successfully!")
    logger.info("=" * 60)
    
    logger.info("🎉 Training pipeline completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Evaluation results: {args.output_dir}/evaluation_results.json")
    
    # Ask if user wants to launch demo
    launch_demo = input("Would you like to launch the demo? (y/n): ").lower().strip()
    
    if launch_demo == 'y':
        logger.info("Launching demo...")
        demo_command = f"python demo.py --model_path {args.output_dir} --port 7860"
        
        logger.info("Demo will be available at: http://localhost:7860")
        logger.info("Press Ctrl+C to stop the demo")
        
        subprocess.run(demo_command, shell=True)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)