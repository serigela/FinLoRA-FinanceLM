"""
Dataset classes for multi-task finance LLM training
"""

import os
import json
import torch
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer
from scripts.utils import setup_logging, load_config, TaskFormatter, load_jsonl

logger = setup_logging()

class FinanceDataset(TorchDataset):
    """Custom dataset for multi-task finance LLM training"""
    
    def __init__(self, 
                 data: List[Dict[str, Any]], 
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 task_type: str = "multi_task"):
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get the formatted text
        text = item.get('text', '')
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        labels = encoding['input_ids'].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten(),
            'task': item.get('task', 'unknown')
        }

class MultiTaskDataLoader:
    """Data loader for multi-task finance datasets"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = load_config(config_path)
        self.training_config = load_config("config/training_config.yaml")
        self.task_formatter = TaskFormatter(self.config)
        
    def load_flare_dataset(self) -> List[Dict[str, Any]]:
        """Load and process FLARE benchmark dataset"""
        logger.info("Loading FLARE dataset...")
        
        try:
            # Load the dataset (you might need to adjust this based on actual FLARE structure)
            dataset = load_dataset("TheFinAI/flare-bench")
            
            processed_data = []
            
            # Process different tasks from FLARE
            for split in ['train', 'validation']:
                if split in dataset:
                    for item in dataset[split]:
                        task_type = item.get('task', 'sentiment')
                        
                        if task_type == 'sentiment':
                            text = self.task_formatter.format_sentiment(
                                item['text'], 
                                item['label']
                            )
                        elif task_type == 'headline':
                            text = self.task_formatter.format_headline(
                                item['text'], 
                                item['label']
                            )
                        elif task_type == 'qa':
                            text = self.task_formatter.format_qa(
                                item.get('context', ''),
                                item.get('question', ''),
                                item.get('answer', '')
                            )
                        else:
                            continue
                        
                        processed_data.append({
                            'text': text,
                            'task': task_type,
                            'split': split
                        })
            
            logger.info(f"Loaded {len(processed_data)} samples from FLARE")
            return processed_data
            
        except Exception as e:
            logger.warning(f"Could not load FLARE dataset: {e}")
            return self._create_sample_flare_data()
    
    def load_ectsum_dataset(self) -> List[Dict[str, Any]]:
        """Load and process ECTSum dataset"""
        logger.info("Loading ECTSum dataset...")
        
        try:
            # Load ECTSum dataset
            dataset = load_dataset("ectsum")
            
            processed_data = []
            
            for split in ['train', 'validation']:
                if split in dataset:
                    for item in dataset[split]:
                        text = self.task_formatter.format_summarization(
                            item['transcript'],
                            item['summary']
                        )
                        
                        processed_data.append({
                            'text': text,
                            'task': 'summarization',
                            'split': split
                        })
            
            logger.info(f"Loaded {len(processed_data)} samples from ECTSum")
            return processed_data
            
        except Exception as e:
            logger.warning(f"Could not load ECTSum dataset: {e}")
            return self._create_sample_ectsum_data()
    
    def load_custom_qa_dataset(self) -> List[Dict[str, Any]]:
        """Load custom QA dataset"""
        logger.info("Loading custom QA dataset...")
        
        qa_path = self.training_config['datasets']['custom_qa']['path']
        
        if os.path.exists(qa_path):
            qa_data = load_jsonl(qa_path)
            
            processed_data = []
            for item in qa_data:
                text = self.task_formatter.format_qa(
                    item['context'],
                    item['question'],
                    item['answer']
                )
                
                processed_data.append({
                    'text': text,
                    'task': 'qa',
                    'split': item.get('split', 'train')
                })
            
            logger.info(f"Loaded {len(processed_data)} samples from custom QA")
            return processed_data
        else:
            logger.warning(f"Custom QA file not found: {qa_path}")
            return self._create_sample_qa_data()
    
    def load_stock_movement_dataset(self) -> List[Dict[str, Any]]:
        """Load stock movement dataset"""
        logger.info("Loading stock movement dataset...")
        
        stock_path = self.training_config['datasets']['stock_data']['path']
        
        if os.path.exists(stock_path):
            stock_data = load_jsonl(stock_path)
            
            processed_data = []
            for item in stock_data:
                text = self.task_formatter.format_regression(
                    item['text'],
                    item['movement']
                )
                
                processed_data.append({
                    'text': text,
                    'task': 'regression',
                    'split': item.get('split', 'train')
                })
            
            logger.info(f"Loaded {len(processed_data)} samples from stock movement")
            return processed_data
        else:
            logger.warning(f"Stock movement file not found: {stock_path}")
            return self._create_sample_stock_data()
    
    def combine_datasets(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Combine all datasets and split into train/validation"""
        logger.info("Combining all datasets...")
        
        # Load all datasets
        flare_data = self.load_flare_dataset()
        ectsum_data = self.load_ectsum_dataset()
        qa_data = self.load_custom_qa_dataset()
        stock_data = self.load_stock_movement_dataset()
        
        # Combine all data
        all_data = flare_data + ectsum_data + qa_data + stock_data
        
        # Split into train/validation
        train_data = [item for item in all_data if item['split'] == 'train']
        val_data = [item for item in all_data if item['split'] == 'validation']
        
        logger.info(f"Combined dataset: {len(train_data)} train, {len(val_data)} validation")
        
        return train_data, val_data
    
    def create_datasets(self, tokenizer: AutoTokenizer) -> Tuple[FinanceDataset, FinanceDataset]:
        """Create training and validation datasets"""
        train_data, val_data = self.combine_datasets()
        
        max_length = self.config['tokenizer']['max_length']
        
        train_dataset = FinanceDataset(train_data, tokenizer, max_length)
        val_dataset = FinanceDataset(val_data, tokenizer, max_length)
        
        return train_dataset, val_dataset
    
    def _create_sample_flare_data(self) -> List[Dict[str, Any]]:
        """Create sample FLARE data for testing"""
        logger.info("Creating sample FLARE data...")
        
        sample_data = []
        
        # Sample sentiment data
        sentiments = [
            ("The company reported strong quarterly earnings.", "positive"),
            ("Stock prices fell significantly after the announcement.", "negative"),
            ("The market remained stable throughout the day.", "neutral"),
            ("Revenue exceeded expectations by 15%.", "positive"),
            ("The company is facing regulatory challenges.", "negative")
        ]
        
        for text, label in sentiments:
            formatted_text = self.task_formatter.format_sentiment(text, label)
            sample_data.append({
                'text': formatted_text,
                'task': 'sentiment',
                'split': 'train'
            })
        
        # Sample headline data
        headlines = [
            ("Tech stocks surge on positive earnings report", "bullish"),
            ("Banking sector faces headwinds", "bearish"),
            ("Market volatility expected to continue", "neutral"),
            ("New product launch drives stock higher", "bullish"),
            ("Company announces layoffs", "bearish")
        ]
        
        for text, label in headlines:
            formatted_text = self.task_formatter.format_headline(text, label)
            sample_data.append({
                'text': formatted_text,
                'task': 'headline',
                'split': 'train'
            })
        
        return sample_data
    
    def _create_sample_ectsum_data(self) -> List[Dict[str, Any]]:
        """Create sample ECTSum data for testing"""
        logger.info("Creating sample ECTSum data...")
        
        sample_data = []
        
        transcripts = [
            {
                'transcript': "Good morning everyone. We're pleased to report that our Q3 results exceeded expectations. Revenue grew 12% year-over-year to $2.1 billion. Our core business segments showed strong performance, particularly in the technology division which grew 18%.",
                'summary': "• Q3 revenue of $2.1B, up 12% YoY\n• Technology division grew 18%\n• Results exceeded expectations"
            },
            {
                'transcript': "Thank you for joining our earnings call. This quarter we faced some challenges in our international markets due to currency fluctuations. However, our domestic business remained strong with 8% growth. We're implementing cost reduction measures to improve margins.",
                'summary': "• International markets challenged by currency issues\n• Domestic business up 8%\n• Implementing cost reduction measures"
            }
        ]
        
        for item in transcripts:
            formatted_text = self.task_formatter.format_summarization(
                item['transcript'], 
                item['summary']
            )
            sample_data.append({
                'text': formatted_text,
                'task': 'summarization',
                'split': 'train'
            })
        
        return sample_data
    
    def _create_sample_qa_data(self) -> List[Dict[str, Any]]:
        """Create sample QA data for testing"""
        logger.info("Creating sample QA data...")
        
        sample_data = []
        
        qa_pairs = [
            {
                'context': "Apple Inc. reported revenue of $117.2 billion for Q1 2024, representing a 2% increase from the previous year. iPhone sales accounted for 52% of total revenue.",
                'question': "What was Apple's Q1 2024 revenue?",
                'answer': "$117.2 billion"
            },
            {
                'context': "The Federal Reserve raised interest rates by 0.25 percentage points to combat inflation. This marks the third rate increase this year.",
                'question': "By how much did the Fed raise rates?",
                'answer': "0.25 percentage points"
            }
        ]
        
        for item in qa_pairs:
            formatted_text = self.task_formatter.format_qa(
                item['context'],
                item['question'],
                item['answer']
            )
            sample_data.append({
                'text': formatted_text,
                'task': 'qa',
                'split': 'train'
            })
        
        return sample_data
    
    def _create_sample_stock_data(self) -> List[Dict[str, Any]]:
        """Create sample stock movement data for testing"""
        logger.info("Creating sample stock movement data...")
        
        sample_data = []
        
        stock_movements = [
            ("Positive earnings report boosts investor confidence", 2.5),
            ("Company announces major acquisition", 4.2),
            ("Regulatory concerns weigh on stock price", -1.8),
            ("Strong quarterly results exceed expectations", 3.1),
            ("Market volatility affects all tech stocks", -0.5)
        ]
        
        for text, movement in stock_movements:
            formatted_text = self.task_formatter.format_regression(text, movement)
            sample_data.append({
                'text': formatted_text,
                'task': 'regression',
                'split': 'train'
            })
        
        return sample_data

if __name__ == "__main__":
    # Test dataset loading
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data loader
    data_loader = MultiTaskDataLoader()
    
    # Create datasets
    train_dataset, val_dataset = data_loader.create_datasets(tokenizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test a sample
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Task: {sample['task']}")
    
    # Decode sample
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"Decoded text: {decoded[:200]}...")