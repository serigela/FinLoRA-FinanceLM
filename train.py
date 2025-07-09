#!/usr/bin/env python3
"""
Main training script for Finance LLM LoRA fine-tuning
"""

import os
import argparse
import torch
import wandb
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import numpy as np
from typing import Dict, Any

from src.models.lora_config import FinanceLLMLoRA
from src.data.datasets import MultiTaskDataLoader
from scripts.utils import setup_logging, load_config, ensure_directories

logger = setup_logging()

class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task learning"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = kwargs.get('task_weights', {})
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with task weighting"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # Get task information
        tasks = inputs.get("task", ["unknown"] * len(labels))
        
        # Compute standard causal LM loss
        loss = outputs.loss
        
        # Apply task-specific weighting if available
        if self.task_weights and isinstance(tasks, list):
            # This is a simplified approach - in practice, you might want
            # more sophisticated task weighting
            task_weights = torch.tensor([
                self.task_weights.get(task, 1.0) for task in tasks
            ]).to(loss.device)
            loss = loss * task_weights.mean()
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    
    # For language modeling, we compute perplexity
    # Remove -100 labels (padding)
    valid_labels = labels[labels != -100]
    valid_predictions = predictions[labels != -100]
    
    # Compute perplexity
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(valid_predictions), 
        torch.tensor(valid_labels)
    )
    perplexity = torch.exp(loss).item()
    
    return {
        "perplexity": perplexity,
        "eval_loss": loss.item()
    }

def main():
    parser = argparse.ArgumentParser(description="Train Finance LLM with LoRA")
    parser.add_argument("--model_config", default="config/model_config.yaml", 
                       help="Path to model configuration")
    parser.add_argument("--training_config", default="config/training_config.yaml",
                       help="Path to training configuration")
    parser.add_argument("--output_dir", default="./output/finlora",
                       help="Output directory for model")
    parser.add_argument("--resume_from_checkpoint", default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with smaller dataset")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.info("Starting Finance LLM LoRA training...")
    logger.info(f"Arguments: {args}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Load configurations
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=training_config['wandb']['project'],
            name=training_config['wandb']['name'],
            config={**model_config, **training_config}
        )
    
    # Create model and tokenizer
    logger.info("Setting up model and tokenizer...")
    lora_model = FinanceLLMLoRA(args.model_config)
    model, tokenizer = lora_model.setup_model_and_tokenizer()
    
    # Print model info
    model_info = lora_model.get_model_info()
    logger.info(f"Model Info: {model_info}")
    
    # Create datasets
    logger.info("Loading datasets...")
    data_loader = MultiTaskDataLoader(args.model_config)
    train_dataset, val_dataset = data_loader.create_datasets(tokenizer)
    
    # Debug mode: use smaller dataset
    if args.debug:
        logger.info("Debug mode: using smaller dataset")
        train_dataset.data = train_dataset.data[:100]
        val_dataset.data = val_dataset.data[:50]
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=training_config['training']['num_train_epochs'],
        per_device_train_batch_size=training_config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['training']['gradient_accumulation_steps'],
        learning_rate=training_config['training']['learning_rate'],
        weight_decay=training_config['training']['weight_decay'],
        warmup_steps=training_config['training']['warmup_steps'],
        logging_steps=training_config['training']['logging_steps'],
        evaluation_strategy=training_config['training']['evaluation_strategy'],
        eval_steps=training_config['training']['eval_steps'],
        save_steps=training_config['training']['save_steps'],
        save_total_limit=training_config['training']['save_total_limit'],
        load_best_model_at_end=training_config['training']['load_best_model_at_end'],
        metric_for_best_model=training_config['training']['metric_for_best_model'],
        greater_is_better=training_config['training']['greater_is_better'],
        fp16=training_config['optimization']['fp16'],
        gradient_checkpointing=training_config['optimization']['gradient_checkpointing'],
        dataloader_num_workers=training_config['optimization']['dataloader_num_workers'],
        remove_unused_columns=training_config['optimization']['remove_unused_columns'],
        optim=training_config['optimization']['optim'],
        lr_scheduler_type=training_config['optimization']['lr_scheduler_type'],
        max_grad_norm=training_config['optimization']['max_grad_norm'],
        report_to=["wandb"] if args.use_wandb else [],
        push_to_hub=training_config['training']['push_to_hub'],
        hub_model_id=training_config['training']['hub_model_id'],
        seed=training_config['data']['seed'],
        resume_from_checkpoint=args.resume_from_checkpoint,
        run_name=training_config['wandb']['name'] if args.use_wandb else None,
    )
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        task_weights=training_config['multi_task']['task_weights']
    )
    
    # Train the model
    logger.info("Starting training...")
    
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Log training results
        logger.info(f"Training completed!")
        logger.info(f"Training loss: {train_result.training_loss}")
        logger.info(f"Training steps: {train_result.global_step}")
        
        # Save the model
        logger.info("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training state
        trainer.save_state()
        
        # Evaluate the model
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        
        logger.info(f"Final evaluation results: {eval_result}")
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "training_metrics.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump({
                "train_loss": train_result.training_loss,
                "eval_results": eval_result,
                "model_info": model_info,
                "training_args": training_args.to_dict()
            }, f, indent=2)
        
        # Optional: merge and save full model
        if training_config['training'].get('save_merged_model', False):
            merged_output_dir = os.path.join(args.output_dir, "merged")
            logger.info(f"Merging and saving full model to {merged_output_dir}")
            lora_model.merge_and_save(merged_output_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()