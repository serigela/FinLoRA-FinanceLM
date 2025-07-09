#!/usr/bin/env python3
"""
Evaluation script for Finance LLM LoRA model
"""

import os
import argparse
import torch
import json
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from src.models.lora_config import FinanceLLMLoRA
from src.data.datasets import MultiTaskDataLoader
from scripts.utils import setup_logging, load_config, parse_instruction_template

logger = setup_logging()

class FinanceLLMEvaluator:
    """Evaluator for Finance LLM LoRA model"""
    
    def __init__(self, model_path: str, config_path: str = "config/model_config.yaml"):
        self.model_path = model_path
        self.config = load_config(config_path)
        self.lora_model = FinanceLLMLoRA(config_path)
        self.model, self.tokenizer = self.lora_model.load_model(model_path)
        self.model.eval()
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract only the generated part
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_response[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        
        return generated_part.strip()
    
    def evaluate_sentiment(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate sentiment classification task"""
        logger.info("Evaluating sentiment classification...")
        
        predictions = []
        targets = []
        
        for item in test_data:
            if item['task'] != 'sentiment':
                continue
                
            # Parse the instruction template
            parsed = parse_instruction_template(item['text'])
            
            # Create prompt without the response
            prompt = f"Instruction: {parsed['instruction']}\nInput: {parsed['input']}\nResponse:"
            
            # Generate prediction
            pred = self.generate_response(prompt, max_length=50)
            predictions.append(pred.lower().strip())
            targets.append(parsed['response'].lower().strip())
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            "sentiment_accuracy": accuracy,
            "sentiment_f1": f1,
            "sentiment_samples": len(targets)
        }
    
    def evaluate_headline(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate headline classification task"""
        logger.info("Evaluating headline classification...")
        
        predictions = []
        targets = []
        
        for item in test_data:
            if item['task'] != 'headline':
                continue
                
            # Parse the instruction template
            parsed = parse_instruction_template(item['text'])
            
            # Create prompt without the response
            prompt = f"Instruction: {parsed['instruction']}\nInput: {parsed['input']}\nResponse:"
            
            # Generate prediction
            pred = self.generate_response(prompt, max_length=50)
            predictions.append(pred.lower().strip())
            targets.append(parsed['response'].lower().strip())
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            "headline_accuracy": accuracy,
            "headline_f1": f1,
            "headline_samples": len(targets)
        }
    
    def evaluate_summarization(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate summarization task"""
        logger.info("Evaluating summarization...")
        
        predictions = []
        targets = []
        
        for item in test_data:
            if item['task'] != 'summarization':
                continue
                
            # Parse the instruction template
            parsed = parse_instruction_template(item['text'])
            
            # Create prompt without the response
            prompt = f"Instruction: {parsed['instruction']}\nInput: {parsed['input']}\nResponse:"
            
            # Generate prediction
            pred = self.generate_response(prompt, max_length=256)
            predictions.append(pred)
            targets.append(parsed['response'])
        
        if not predictions:
            return {"summarization_samples": 0}
        
        # Calculate ROUGE scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, target in zip(predictions, targets):
            scores = self.rouge_scorer.score(target, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Calculate BERTScore
        try:
            P, R, F1 = bert_score(predictions, targets, lang='en', verbose=False)
            bert_score_f1 = F1.mean().item()
        except:
            bert_score_f1 = 0.0
        
        return {
            "summarization_rouge1": np.mean(rouge_scores['rouge1']),
            "summarization_rouge2": np.mean(rouge_scores['rouge2']),
            "summarization_rougeL": np.mean(rouge_scores['rougeL']),
            "summarization_bert_score": bert_score_f1,
            "summarization_samples": len(predictions)
        }
    
    def evaluate_qa(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate QA task"""
        logger.info("Evaluating QA...")
        
        predictions = []
        targets = []
        
        for item in test_data:
            if item['task'] != 'qa':
                continue
                
            # Parse the instruction template
            parsed = parse_instruction_template(item['text'])
            
            # Create prompt without the response
            prompt = f"Instruction: {parsed['instruction']}\nInput: {parsed['input']}\nResponse:"
            
            # Generate prediction
            pred = self.generate_response(prompt, max_length=128)
            predictions.append(pred)
            targets.append(parsed['response'])
        
        if not predictions:
            return {"qa_samples": 0}
        
        # Calculate exact match and F1 (simplified for now)
        exact_matches = sum(1 for p, t in zip(predictions, targets) if p.strip().lower() == t.strip().lower())
        exact_match_score = exact_matches / len(predictions)
        
        # Calculate ROUGE F1 as a proxy for QA F1
        rouge_f1_scores = []
        for pred, target in zip(predictions, targets):
            scores = self.rouge_scorer.score(target, pred)
            rouge_f1_scores.append(scores['rouge1'].fmeasure)
        
        return {
            "qa_exact_match": exact_match_score,
            "qa_f1": np.mean(rouge_f1_scores),
            "qa_samples": len(predictions)
        }
    
    def evaluate_regression(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate regression task"""
        logger.info("Evaluating regression...")
        
        predictions = []
        targets = []
        
        for item in test_data:
            if item['task'] != 'regression':
                continue
                
            # Parse the instruction template
            parsed = parse_instruction_template(item['text'])
            
            # Create prompt without the response
            prompt = f"Instruction: {parsed['instruction']}\nInput: {parsed['input']}\nResponse:"
            
            # Generate prediction
            pred = self.generate_response(prompt, max_length=50)
            
            # Try to extract numerical value
            try:
                pred_value = float(pred.strip())
                target_value = float(parsed['response'].strip())
                
                predictions.append(pred_value)
                targets.append(target_value)
            except:
                continue
        
        if not predictions:
            return {"regression_samples": 0}
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return {
            "regression_mse": mse,
            "regression_r2": r2,
            "regression_samples": len(predictions)
        }
    
    def evaluate_all(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all tasks"""
        logger.info("Starting comprehensive evaluation...")
        
        results = {}
        
        # Evaluate each task
        results.update(self.evaluate_sentiment(test_data))
        results.update(self.evaluate_headline(test_data))
        results.update(self.evaluate_summarization(test_data))
        results.update(self.evaluate_qa(test_data))
        results.update(self.evaluate_regression(test_data))
        
        # Calculate overall metrics
        total_samples = sum(v for k, v in results.items() if k.endswith('_samples'))
        results['total_samples'] = total_samples
        
        logger.info("Evaluation completed!")
        return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Finance LLM LoRA model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--config_path", default="config/model_config.yaml", 
                       help="Path to model configuration")
    parser.add_argument("--output_file", default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--use_test_data", action="store_true",
                       help="Use test split instead of validation")
    
    args = parser.parse_args()
    
    logger.info(f"Starting evaluation of model: {args.model_path}")
    
    # Create evaluator
    evaluator = FinanceLLMEvaluator(args.model_path, args.config_path)
    
    # Load test data
    data_loader = MultiTaskDataLoader(args.config_path)
    train_data, val_data = data_loader.combine_datasets()
    
    # Use validation data for evaluation (or test data if available)
    test_data = val_data  # In practice, you'd want a separate test set
    
    logger.info(f"Evaluating on {len(test_data)} samples")
    
    # Run evaluation
    results = evaluator.evaluate_all(test_data)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    
    for task in ['sentiment', 'headline', 'summarization', 'qa', 'regression']:
        logger.info(f"\n{task.upper()} TASK:")
        task_results = {k: v for k, v in results.items() if k.startswith(task)}
        for metric, value in task_results.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    
    logger.info(f"\nTotal samples evaluated: {results['total_samples']}")
    logger.info(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()