#!/usr/bin/env python3
"""
Preprocessing script for finance datasets
"""

import os
import argparse
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils import setup_logging, load_config, load_jsonl, save_jsonl, TaskFormatter

logger = setup_logging()

class FinanceDataPreprocessor:
    """Preprocesses finance datasets into instruction format"""
    
    def __init__(self, config_path: str = "../config/model_config.yaml"):
        self.config = load_config(config_path)
        self.task_formatter = TaskFormatter(self.config)
        self.data_dir = "./data"
        self.output_dir = "./data/processed"
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def preprocess_flare_data(self) -> List[Dict[str, Any]]:
        """Preprocess FLARE benchmark data"""
        logger.info("Preprocessing FLARE data...")
        
        flare_file = os.path.join(self.data_dir, "flare_data.jsonl")
        
        if not os.path.exists(flare_file):
            logger.error(f"FLARE data file not found: {flare_file}")
            return []
        
        raw_data = load_jsonl(flare_file)
        processed_data = []
        
        for item in raw_data:
            task_type = item.get('task', 'sentiment')
            
            if task_type == 'sentiment':
                formatted_text = self.task_formatter.format_sentiment(
                    item['text'], 
                    item['label']
                )
            elif task_type == 'headline':
                formatted_text = self.task_formatter.format_headline(
                    item['text'], 
                    item['label']
                )
            else:
                continue
            
            processed_data.append({
                'text': formatted_text,
                'task': task_type,
                'split': item.get('split', 'train')
            })
        
        logger.info(f"Processed {len(processed_data)} FLARE samples")
        return processed_data
    
    def preprocess_ectsum_data(self) -> List[Dict[str, Any]]:
        """Preprocess ECTSum data"""
        logger.info("Preprocessing ECTSum data...")
        
        ectsum_file = os.path.join(self.data_dir, "ectsum_data.jsonl")
        
        if not os.path.exists(ectsum_file):
            logger.error(f"ECTSum data file not found: {ectsum_file}")
            return []
        
        raw_data = load_jsonl(ectsum_file)
        processed_data = []
        
        for item in raw_data:
            formatted_text = self.task_formatter.format_summarization(
                item['transcript'],
                item['summary']
            )
            
            processed_data.append({
                'text': formatted_text,
                'task': 'summarization',
                'split': item.get('split', 'train')
            })
        
        logger.info(f"Processed {len(processed_data)} ECTSum samples")
        return processed_data
    
    def preprocess_qa_data(self) -> List[Dict[str, Any]]:
        """Preprocess QA data"""
        logger.info("Preprocessing QA data...")
        
        qa_file = os.path.join(self.data_dir, "custom_qa.jsonl")
        
        if not os.path.exists(qa_file):
            logger.error(f"QA data file not found: {qa_file}")
            return []
        
        raw_data = load_jsonl(qa_file)
        processed_data = []
        
        for item in raw_data:
            formatted_text = self.task_formatter.format_qa(
                item['context'],
                item['question'],
                item['answer']
            )
            
            processed_data.append({
                'text': formatted_text,
                'task': 'qa',
                'split': item.get('split', 'train')
            })
        
        logger.info(f"Processed {len(processed_data)} QA samples")
        return processed_data
    
    def preprocess_stock_data(self) -> List[Dict[str, Any]]:
        """Preprocess stock movement data"""
        logger.info("Preprocessing stock movement data...")
        
        stock_file = os.path.join(self.data_dir, "stock_movement.jsonl")
        
        if not os.path.exists(stock_file):
            logger.error(f"Stock data file not found: {stock_file}")
            return []
        
        raw_data = load_jsonl(stock_file)
        processed_data = []
        
        for item in raw_data:
            formatted_text = self.task_formatter.format_regression(
                item['text'],
                item['movement']
            )
            
            processed_data.append({
                'text': formatted_text,
                'task': 'regression',
                'split': item.get('split', 'train')
            })
        
        logger.info(f"Processed {len(processed_data)} stock movement samples")
        return processed_data
    
    def balance_datasets(self, datasets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Balance datasets for multi-task learning"""
        logger.info("Balancing datasets for multi-task learning...")
        
        # Combine all datasets
        combined_data = []
        for dataset in datasets:
            combined_data.extend(dataset)
        
        # Count samples per task
        task_counts = {}
        for item in combined_data:
            task = item['task']
            task_counts[task] = task_counts.get(task, 0) + 1
        
        logger.info(f"Task distribution: {task_counts}")
        
        # Apply balancing strategy
        training_config = load_config("../config/training_config.yaml")
        strategy = training_config['multi_task']['sampling_strategy']
        
        if strategy == 'balanced':
            # Ensure each task has roughly the same number of samples
            min_samples = min(task_counts.values())
            balanced_data = []
            
            for task in task_counts:
                task_samples = [item for item in combined_data if item['task'] == task]
                # Randomly sample min_samples from each task
                if len(task_samples) > min_samples:
                    import random
                    random.shuffle(task_samples)
                    task_samples = task_samples[:min_samples]
                balanced_data.extend(task_samples)
            
            logger.info(f"Balanced to {len(balanced_data)} samples")
            return balanced_data
        
        elif strategy == 'proportional':
            # Keep original proportions
            return combined_data
        
        else:
            # Custom strategy - apply task weights
            task_weights = training_config['multi_task']['task_weights']
            weighted_data = []
            
            for task in task_counts:
                task_samples = [item for item in combined_data if item['task'] == task]
                weight = task_weights.get(task, 1.0)
                
                # Adjust sample count based on weight
                target_samples = int(len(task_samples) * weight)
                
                if target_samples > len(task_samples):
                    # Oversample
                    import random
                    oversampled = random.choices(task_samples, k=target_samples)
                    weighted_data.extend(oversampled)
                else:
                    # Undersample
                    import random
                    random.shuffle(task_samples)
                    weighted_data.extend(task_samples[:target_samples])
            
            logger.info(f"Weighted to {len(weighted_data)} samples")
            return weighted_data
    
    def create_train_val_split(self, data: List[Dict[str, Any]]) -> tuple:
        """Create train/validation split"""
        logger.info("Creating train/validation split...")
        
        # Separate by existing split if available
        train_data = [item for item in data if item.get('split') == 'train']
        val_data = [item for item in data if item.get('split') == 'validation']
        
        # If no predefined split, create one
        if not val_data:
            logger.info("No validation split found, creating one...")
            
            # Group by task to ensure balanced split
            task_groups = {}
            for item in data:
                task = item['task']
                if task not in task_groups:
                    task_groups[task] = []
                task_groups[task].append(item)
            
            train_data = []
            val_data = []
            
            for task, samples in task_groups.items():
                task_train, task_val = train_test_split(
                    samples, 
                    test_size=0.2, 
                    random_state=42
                )
                
                # Update split labels
                for item in task_train:
                    item['split'] = 'train'
                for item in task_val:
                    item['split'] = 'validation'
                
                train_data.extend(task_train)
                val_data.extend(task_val)
        
        logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
        return train_data, val_data
    
    def preprocess_all(self, tasks: List[str]) -> None:
        """Preprocess all specified tasks"""
        logger.info(f"Preprocessing tasks: {tasks}")
        
        datasets = []
        
        if 'flare' in tasks or 'sentiment' in tasks:
            datasets.append(self.preprocess_flare_data())
        
        if 'ectsum' in tasks or 'summarization' in tasks:
            datasets.append(self.preprocess_ectsum_data())
        
        if 'qa' in tasks:
            datasets.append(self.preprocess_qa_data())
        
        if 'regression' in tasks or 'stock' in tasks:
            datasets.append(self.preprocess_stock_data())
        
        # Filter out empty datasets
        datasets = [d for d in datasets if d]
        
        if not datasets:
            logger.error("No datasets to process!")
            return
        
        # Balance datasets
        balanced_data = self.balance_datasets(datasets)
        
        # Create train/validation split
        train_data, val_data = self.create_train_val_split(balanced_data)
        
        # Save processed data
        train_file = os.path.join(self.output_dir, "train.jsonl")
        val_file = os.path.join(self.output_dir, "validation.jsonl")
        
        save_jsonl(train_data, train_file)
        save_jsonl(val_data, val_file)
        
        logger.info(f"Preprocessing completed!")
        logger.info(f"Train data saved to: {train_file}")
        logger.info(f"Validation data saved to: {val_file}")
        
        # Save statistics
        stats = {
            'total_samples': len(balanced_data),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'task_distribution': {}
        }
        
        for task in ['sentiment', 'headline', 'summarization', 'qa', 'regression']:
            train_count = sum(1 for item in train_data if item['task'] == task)
            val_count = sum(1 for item in val_data if item['task'] == task)
            stats['task_distribution'][task] = {
                'train': train_count,
                'validation': val_count,
                'total': train_count + val_count
            }
        
        stats_file = os.path.join(self.output_dir, "preprocessing_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess finance datasets")
    parser.add_argument("--config", default="../config/model_config.yaml",
                       help="Path to model configuration")
    parser.add_argument("--tasks", nargs='+', 
                       choices=['flare', 'sentiment', 'ectsum', 'summarization', 'qa', 'regression', 'stock'],
                       default=['flare', 'ectsum', 'qa', 'regression'],
                       help="Tasks to preprocess")
    parser.add_argument("--data_dir", default="./data",
                       help="Input data directory")
    parser.add_argument("--output_dir", default="./data/processed",
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    preprocessor = FinanceDataPreprocessor(args.config)
    preprocessor.data_dir = args.data_dir
    preprocessor.output_dir = args.output_dir
    
    preprocessor.preprocess_all(args.tasks)

if __name__ == "__main__":
    main()