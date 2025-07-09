"""
LoRA configuration and model setup for Finance LLM
"""

import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from scripts.utils import setup_logging, load_config, count_parameters, count_trainable_parameters, format_number

logger = setup_logging()

class FinanceLLMLoRA:
    """Finance LLM with LoRA configuration"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        
    def create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration"""
        quant_config = self.config['base_model']['quantization']
        
        if quant_config['load_in_4bit']:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type']
            )
        elif quant_config['load_in_8bit']:
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None
    
    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration"""
        lora_config = self.config['lora_config']
        
        return LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=lora_config.get('modules_to_save', [])
        )
    
    def load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer"""
        model_name = self.config['base_model']['name']
        tokenizer_config = self.config['tokenizer']
        
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Configure tokenizer settings
        tokenizer.padding_side = tokenizer_config['padding_side']
        tokenizer.model_max_length = tokenizer_config['max_length']
        
        return tokenizer
    
    def load_base_model(self) -> AutoModelForCausalLM:
        """Load base model with quantization"""
        model_name = self.config['base_model']['name']
        quantization_config = self.create_quantization_config()
        
        logger.info(f"Loading base model: {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        return model
    
    def apply_lora(self, model: AutoModelForCausalLM) -> PeftModel:
        """Apply LoRA adapters to the model"""
        lora_config = self.create_lora_config()
        
        logger.info("Applying LoRA adapters")
        logger.info(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        logger.info(f"Target modules: {lora_config.target_modules}")
        
        peft_model = get_peft_model(model, lora_config)
        
        # Print parameter counts
        total_params = count_parameters(peft_model)
        trainable_params = count_trainable_parameters(peft_model)
        
        logger.info(f"Total parameters: {format_number(total_params)}")
        logger.info(f"Trainable parameters: {format_number(trainable_params)}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        return peft_model
    
    def setup_model_and_tokenizer(self):
        """Set up the complete model and tokenizer"""
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        
        # Load base model
        base_model = self.load_base_model()
        
        # Apply LoRA
        self.model = self.apply_lora(base_model)
        
        # Prepare model for training
        self.model.train()
        
        return self.model, self.tokenizer
    
    def save_model(self, output_dir: str):
        """Save the LoRA model"""
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load_model(self, model_path: str):
        """Load a trained LoRA model"""
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        base_model = self.load_base_model()
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        return self.model, self.tokenizer
    
    def merge_and_save(self, output_dir: str):
        """Merge LoRA weights with base model and save"""
        logger.info("Merging LoRA weights with base model")
        
        # Merge LoRA weights
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Merged model saved to {output_dir}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        total_params = count_parameters(self.model)
        trainable_params = count_trainable_parameters(self.model)
        
        return {
            "base_model": self.config['base_model']['name'],
            "lora_r": self.config['lora_config']['r'],
            "lora_alpha": self.config['lora_config']['lora_alpha'],
            "target_modules": self.config['lora_config']['target_modules'],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }

def create_model_and_tokenizer(config_path: str = "config/model_config.yaml"):
    """Convenience function to create model and tokenizer"""
    lora_model = FinanceLLMLoRA(config_path)
    return lora_model.setup_model_and_tokenizer()

if __name__ == "__main__":
    # Test model loading
    logger.info("Testing model loading...")
    
    lora_model = FinanceLLMLoRA()
    model, tokenizer = lora_model.setup_model_and_tokenizer()
    
    # Print model info
    info = lora_model.get_model_info()
    print(f"Model Info: {info}")
    
    # Test tokenization
    test_text = "The stock market showed positive sentiment today."
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"Tokenized text shape: {tokens['input_ids'].shape}")
    
    logger.info("Model loading test completed successfully!")