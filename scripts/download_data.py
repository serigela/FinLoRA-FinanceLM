#!/usr/bin/env python3
"""
Script to download and prepare finance datasets
"""

import os
import json
import yfinance as yf
from datasets import load_dataset
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

from utils import setup_logging, load_config, save_jsonl, create_directories

logger = setup_logging()

class FinanceDataDownloader:
    """Downloads and prepares finance datasets"""
    
    def __init__(self, config_path: str = "../config/model_config.yaml"):
        self.config = load_config(config_path)
        self.output_dir = "./data"
        create_directories([self.output_dir])
    
    def download_flare_data(self) -> None:
        """Download FLARE benchmark data"""
        logger.info("Downloading FLARE benchmark data...")
        
        try:
            # Try to load the actual FLARE dataset
            dataset = load_dataset("TheFinAI/flare-bench")
            
            # Process and save
            processed_data = []
            
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    for item in dataset[split]:
                        processed_data.append({
                            'text': item.get('text', ''),
                            'label': item.get('label', ''),
                            'task': item.get('task', 'sentiment'),
                            'split': split
                        })
            
            # Save to JSONL
            save_jsonl(processed_data, os.path.join(self.output_dir, "flare_data.jsonl"))
            logger.info(f"Saved {len(processed_data)} FLARE samples")
            
        except Exception as e:
            logger.warning(f"Could not download FLARE data: {e}")
            self._create_sample_flare_data()
    
    def download_ectsum_data(self) -> None:
        """Download ECTSum earnings call data"""
        logger.info("Downloading ECTSum data...")
        
        try:
            # Try to load ECTSum dataset
            dataset = load_dataset("ectsum")
            
            processed_data = []
            
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    for item in dataset[split]:
                        processed_data.append({
                            'transcript': item.get('transcript', ''),
                            'summary': item.get('summary', ''),
                            'task': 'summarization',
                            'split': split
                        })
            
            # Save to JSONL
            save_jsonl(processed_data, os.path.join(self.output_dir, "ectsum_data.jsonl"))
            logger.info(f"Saved {len(processed_data)} ECTSum samples")
            
        except Exception as e:
            logger.warning(f"Could not download ECTSum data: {e}")
            self._create_sample_ectsum_data()
    
    def create_custom_qa_data(self) -> None:
        """Create custom QA data from financial sources"""
        logger.info("Creating custom QA data...")
        
        # Sample financial QA data
        qa_data = [
            {
                'context': "Apple Inc. (AAPL) reported quarterly revenue of $89.5 billion, beating analyst expectations of $88.9 billion. The company's iPhone sales contributed 52% of total revenue, while Services revenue grew 16% year-over-year.",
                'question': "What was Apple's quarterly revenue?",
                'answer': "$89.5 billion",
                'task': 'qa',
                'split': 'train'
            },
            {
                'context': "Tesla's Q3 2023 earnings showed a 9% increase in vehicle deliveries compared to the previous quarter. The company delivered 435,059 vehicles and reported automotive revenue of $19.625 billion.",
                'question': "How many vehicles did Tesla deliver in Q3 2023?",
                'answer': "435,059 vehicles",
                'task': 'qa',
                'split': 'train'
            },
            {
                'context': "Microsoft's cloud revenue from Azure and other cloud services increased 29% in the most recent quarter. The company's total revenue for the quarter was $52.7 billion.",
                'question': "What was Microsoft's total quarterly revenue?",
                'answer': "$52.7 billion",
                'task': 'qa',
                'split': 'validation'
            },
            {
                'context': "Amazon Web Services (AWS) reported $23.06 billion in revenue for Q3, representing a 12% year-over-year growth. This segment accounts for approximately 70% of Amazon's operating income.",
                'question': "What was AWS revenue for Q3?",
                'answer': "$23.06 billion",
                'task': 'qa',
                'split': 'validation'
            },
            {
                'context': "The Federal Reserve raised the federal funds rate by 0.75 percentage points to a range of 3.75% to 4.00%. This was the fourth consecutive rate hike as the Fed continues to combat inflation.",
                'question': "What is the current federal funds rate range?",
                'answer': "3.75% to 4.00%",
                'task': 'qa',
                'split': 'train'
            }
        ]
        
        # Save to JSONL
        save_jsonl(qa_data, os.path.join(self.output_dir, "custom_qa.jsonl"))
        logger.info(f"Created {len(qa_data)} custom QA samples")
    
    def create_stock_movement_data(self) -> None:
        """Create stock movement data using yfinance"""
        logger.info("Creating stock movement data...")
        
        # Stock symbols to analyze
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
        
        stock_data = []
        
        for symbol in symbols:
            try:
                # Get stock data
                stock = yf.Ticker(symbol)
                
                # Get recent news and price data
                news = stock.news[:5]  # Get last 5 news items
                hist = stock.history(period="1mo")
                
                for news_item in news:
                    try:
                        # Calculate price movement around news date
                        news_date = datetime.fromtimestamp(news_item['providerPublishTime'])
                        
                        # Find closest trading day
                        closest_date = None
                        min_diff = float('inf')
                        
                        for date in hist.index:
                            diff = abs((date.date() - news_date.date()).days)
                            if diff < min_diff:
                                min_diff = diff
                                closest_date = date
                        
                        if closest_date is not None and min_diff <= 2:  # Within 2 days
                            # Calculate percentage change
                            idx = hist.index.get_loc(closest_date)
                            if idx < len(hist) - 1:
                                current_price = hist.iloc[idx]['Close']
                                next_price = hist.iloc[idx + 1]['Close']
                                pct_change = ((next_price - current_price) / current_price) * 100
                                
                                # Create training sample
                                stock_data.append({
                                    'text': f"{symbol}: {news_item['title']}",
                                    'movement': round(pct_change, 2),
                                    'symbol': symbol,
                                    'date': news_date.strftime('%Y-%m-%d'),
                                    'task': 'regression',
                                    'split': 'train' if len(stock_data) % 5 != 0 else 'validation'
                                })
                    
                    except Exception as e:
                        logger.warning(f"Error processing news for {symbol}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {e}")
                continue
        
        # Add some sample data if real data is limited
        if len(stock_data) < 20:
            sample_data = [
                {
                    'text': 'AAPL: Apple announces new iPhone with advanced AI features',
                    'movement': 2.5,
                    'symbol': 'AAPL',
                    'date': '2024-01-15',
                    'task': 'regression',
                    'split': 'train'
                },
                {
                    'text': 'MSFT: Microsoft beats earnings expectations with strong cloud growth',
                    'movement': 3.2,
                    'symbol': 'MSFT',
                    'date': '2024-01-20',
                    'task': 'regression',
                    'split': 'train'
                },
                {
                    'text': 'TSLA: Tesla recalls vehicles due to safety concerns',
                    'movement': -1.8,
                    'symbol': 'TSLA',
                    'date': '2024-01-25',
                    'task': 'regression',
                    'split': 'validation'
                }
            ]
            stock_data.extend(sample_data)
        
        # Save to JSONL
        save_jsonl(stock_data, os.path.join(self.output_dir, "stock_movement.jsonl"))
        logger.info(f"Created {len(stock_data)} stock movement samples")
    
    def _create_sample_flare_data(self) -> None:
        """Create sample FLARE data"""
        logger.info("Creating sample FLARE data...")
        
        sample_data = [
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
            }
        ]
        
        save_jsonl(sample_data, os.path.join(self.output_dir, "flare_data.jsonl"))
        logger.info(f"Created {len(sample_data)} sample FLARE samples")
    
    def _create_sample_ectsum_data(self) -> None:
        """Create sample ECTSum data"""
        logger.info("Creating sample ECTSum data...")
        
        sample_data = [
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
        
        save_jsonl(sample_data, os.path.join(self.output_dir, "ectsum_data.jsonl"))
        logger.info(f"Created {len(sample_data)} sample ECTSum samples")
    
    def download_all(self) -> None:
        """Download all datasets"""
        logger.info("Starting data download process...")
        
        self.download_flare_data()
        self.download_ectsum_data()
        self.create_custom_qa_data()
        self.create_stock_movement_data()
        
        logger.info("Data download completed!")
        logger.info(f"All data saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download finance datasets")
    parser.add_argument("--config", default="../config/model_config.yaml",
                       help="Path to model configuration")
    parser.add_argument("--output_dir", default="./data",
                       help="Output directory for data")
    parser.add_argument("--datasets", nargs='+', 
                       choices=['flare', 'ectsum', 'qa', 'stock', 'all'],
                       default=['all'],
                       help="Datasets to download")
    
    args = parser.parse_args()
    
    downloader = FinanceDataDownloader(args.config)
    downloader.output_dir = args.output_dir
    
    if 'all' in args.datasets:
        downloader.download_all()
    else:
        if 'flare' in args.datasets:
            downloader.download_flare_data()
        if 'ectsum' in args.datasets:
            downloader.download_ectsum_data()
        if 'qa' in args.datasets:
            downloader.create_custom_qa_data()
        if 'stock' in args.datasets:
            downloader.create_stock_movement_data()
    
    logger.info("Download process completed!")

if __name__ == "__main__":
    main()