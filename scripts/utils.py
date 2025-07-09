"""
Utility functions for the Finance LLM LoRA project
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def create_directories(paths: List[str]) -> None:
    """Create directories if they don't exist"""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def get_model_cache_dir() -> str:
    """Get model cache directory from environment"""
    return os.getenv('MODEL_CACHE_DIR', './models')

def get_data_cache_dir() -> str:
    """Get data cache directory from environment"""
    return os.getenv('DATA_CACHE_DIR', './data')

def get_output_dir() -> str:
    """Get output directory from environment"""
    return os.getenv('OUTPUT_DIR', './output')

def format_instruction_template(instruction: str, input_text: str, output_text: str) -> str:
    """Format data into instruction template"""
    return f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"

def parse_instruction_template(text: str) -> Dict[str, str]:
    """Parse instruction template back to components"""
    parts = text.split('\n')
    result = {}
    
    for part in parts:
        if part.startswith('Instruction: '):
            result['instruction'] = part[13:]  # Remove 'Instruction: '
        elif part.startswith('Input: '):
            result['input'] = part[7:]  # Remove 'Input: '
        elif part.startswith('Response: '):
            result['response'] = part[10:]  # Remove 'Response: '
    
    return result

def count_parameters(model) -> int:
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_number(num: int) -> str:
    """Format large numbers in a readable way"""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        get_model_cache_dir(),
        get_data_cache_dir(),
        get_output_dir(),
        './logs',
        './checkpoints'
    ]
    create_directories(directories)

class TaskFormatter:
    """Class to handle task-specific formatting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tasks = config.get('tasks', {})
    
    def format_sentiment(self, text: str, label: str) -> str:
        """Format sentiment classification task"""
        instruction = self.tasks['sentiment']['instruction']
        return format_instruction_template(instruction, text, label)
    
    def format_headline(self, text: str, label: str) -> str:
        """Format headline classification task"""
        instruction = self.tasks['headline']['instruction']
        return format_instruction_template(instruction, text, label)
    
    def format_summarization(self, text: str, summary: str) -> str:
        """Format summarization task"""
        instruction = self.tasks['summarization']['instruction']
        return format_instruction_template(instruction, text, summary)
    
    def format_qa(self, context: str, question: str, answer: str) -> str:
        """Format QA task"""
        instruction = self.tasks['qa']['instruction']
        input_text = f"Context: {context}\nQuestion: {question}"
        return format_instruction_template(instruction, input_text, answer)
    
    def format_regression(self, text: str, value: float) -> str:
        """Format regression task"""
        instruction = self.tasks['regression']['instruction']
        return format_instruction_template(instruction, text, str(value))