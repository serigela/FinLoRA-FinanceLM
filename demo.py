#!/usr/bin/env python3
"""
Gradio demo for Finance LLM LoRA model
"""

import gradio as gr
import torch
from typing import Dict, Any
import os
import argparse

from src.models.lora_config import FinanceLLMLoRA
from scripts.utils import setup_logging, load_config

logger = setup_logging()

class FinanceLLMDemo:
    """Demo interface for Finance LLM"""
    
    def __init__(self, model_path: str, config_path: str = "config/model_config.yaml"):
        self.config = load_config(config_path)
        self.model_path = model_path
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        self.lora_model = FinanceLLMLoRA(config_path)
        self.model, self.tokenizer = self.lora_model.load_model(model_path)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate response from the model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract only the generated part
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = full_response[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
            
            return generated_part.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def sentiment_analysis(self, text: str) -> str:
        """Perform sentiment analysis"""
        instruction = self.config['tasks']['sentiment']['instruction']
        prompt = f"Instruction: {instruction}\nInput: {text}\nResponse:"
        
        return self.generate_response(prompt, max_length=100)
    
    def headline_classification(self, text: str) -> str:
        """Perform headline classification"""
        instruction = self.config['tasks']['headline']['instruction']
        prompt = f"Instruction: {instruction}\nInput: {text}\nResponse:"
        
        return self.generate_response(prompt, max_length=100)
    
    def summarization(self, text: str) -> str:
        """Perform summarization"""
        instruction = self.config['tasks']['summarization']['instruction']
        prompt = f"Instruction: {instruction}\nInput: {text}\nResponse:"
        
        return self.generate_response(prompt, max_length=300)
    
    def financial_qa(self, context: str, question: str) -> str:
        """Answer financial questions"""
        instruction = self.config['tasks']['qa']['instruction']
        input_text = f"Context: {context}\nQuestion: {question}"
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        
        return self.generate_response(prompt, max_length=200)
    
    def stock_prediction(self, text: str) -> str:
        """Predict stock movement"""
        instruction = self.config['tasks']['regression']['instruction']
        prompt = f"Instruction: {instruction}\nInput: {text}\nResponse:"
        
        return self.generate_response(prompt, max_length=100)
    
    def create_demo(self) -> gr.Blocks:
        """Create Gradio demo interface"""
        
        with gr.Blocks(title="Finance LLM Demo", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🏦 Finance LLM with LoRA Demo")
            gr.Markdown("This demo showcases a multi-task finance LLM fine-tuned with LoRA adapters.")
            
            with gr.Tabs():
                # Sentiment Analysis Tab
                with gr.TabItem("💭 Sentiment Analysis"):
                    gr.Markdown("Analyze the sentiment of financial statements.")
                    
                    with gr.Row():
                        with gr.Column():
                            sentiment_input = gr.Textbox(
                                label="Financial Statement",
                                placeholder="Enter a financial statement or news...",
                                lines=3
                            )
                            sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
                        
                        with gr.Column():
                            sentiment_output = gr.Textbox(
                                label="Sentiment",
                                lines=2
                            )
                    
                    sentiment_btn.click(
                        self.sentiment_analysis,
                        inputs=[sentiment_input],
                        outputs=[sentiment_output]
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["The company reported record quarterly earnings with a 25% increase in revenue."],
                            ["Stock prices plummeted after the disappointing earnings announcement."],
                            ["The market remained stable following the Federal Reserve's decision."]
                        ],
                        inputs=[sentiment_input]
                    )
                
                # Headline Classification Tab
                with gr.TabItem("📰 Headline Classification"):
                    gr.Markdown("Classify financial headlines as bullish, bearish, or neutral.")
                    
                    with gr.Row():
                        with gr.Column():
                            headline_input = gr.Textbox(
                                label="Financial Headline",
                                placeholder="Enter a financial headline...",
                                lines=2
                            )
                            headline_btn = gr.Button("Classify Headline", variant="primary")
                        
                        with gr.Column():
                            headline_output = gr.Textbox(
                                label="Classification",
                                lines=2
                            )
                    
                    headline_btn.click(
                        self.headline_classification,
                        inputs=[headline_input],
                        outputs=[headline_output]
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["Tech stocks surge on AI breakthrough announcement"],
                            ["Banking sector faces regulatory headwinds"],
                            ["Market volatility expected to continue"]
                        ],
                        inputs=[headline_input]
                    )
                
                # Summarization Tab
                with gr.TabItem("📄 Earnings Summarization"):
                    gr.Markdown("Summarize earnings call transcripts into key bullet points.")
                    
                    with gr.Row():
                        with gr.Column():
                            summary_input = gr.Textbox(
                                label="Earnings Call Transcript",
                                placeholder="Enter earnings call transcript...",
                                lines=6
                            )
                            summary_btn = gr.Button("Summarize", variant="primary")
                        
                        with gr.Column():
                            summary_output = gr.Textbox(
                                label="Summary",
                                lines=6
                            )
                    
                    summary_btn.click(
                        self.summarization,
                        inputs=[summary_input],
                        outputs=[summary_output]
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["Good morning everyone. We're pleased to report that our Q3 results exceeded expectations. Revenue grew 12% year-over-year to $2.1 billion. Our core business segments showed strong performance, particularly in the technology division which grew 18%. We're maintaining our full-year guidance and expect continued growth in Q4."]
                        ],
                        inputs=[summary_input]
                    )
                
                # Financial Q&A Tab
                with gr.TabItem("❓ Financial Q&A"):
                    gr.Markdown("Ask questions about financial contexts and get answers.")
                    
                    with gr.Row():
                        with gr.Column():
                            qa_context = gr.Textbox(
                                label="Financial Context",
                                placeholder="Enter financial context...",
                                lines=4
                            )
                            qa_question = gr.Textbox(
                                label="Question",
                                placeholder="Enter your question...",
                                lines=2
                            )
                            qa_btn = gr.Button("Get Answer", variant="primary")
                        
                        with gr.Column():
                            qa_output = gr.Textbox(
                                label="Answer",
                                lines=4
                            )
                    
                    qa_btn.click(
                        self.financial_qa,
                        inputs=[qa_context, qa_question],
                        outputs=[qa_output]
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            [
                                "Apple Inc. reported revenue of $117.2 billion for Q1 2024, representing a 2% increase from the previous year. iPhone sales accounted for 52% of total revenue.",
                                "What was Apple's Q1 2024 revenue?"
                            ]
                        ],
                        inputs=[qa_context, qa_question]
                    )
                
                # Stock Prediction Tab
                with gr.TabItem("📈 Stock Movement Prediction"):
                    gr.Markdown("Predict stock price movement based on news sentiment.")
                    
                    with gr.Row():
                        with gr.Column():
                            stock_input = gr.Textbox(
                                label="Stock News",
                                placeholder="Enter stock-related news...",
                                lines=3
                            )
                            stock_btn = gr.Button("Predict Movement", variant="primary")
                        
                        with gr.Column():
                            stock_output = gr.Textbox(
                                label="Predicted Movement (%)",
                                lines=2
                            )
                    
                    stock_btn.click(
                        self.stock_prediction,
                        inputs=[stock_input],
                        outputs=[stock_output]
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["AAPL: Apple announces breakthrough in AI chip technology"],
                            ["TSLA: Tesla reports record quarterly deliveries"],
                            ["MSFT: Microsoft faces antitrust investigation"]
                        ],
                        inputs=[stock_input]
                    )
            
            gr.Markdown("---")
            gr.Markdown("*This demo uses a finance LLM fine-tuned with LoRA adapters for efficient multi-task learning.*")
        
        return demo

def main():
    parser = argparse.ArgumentParser(description="Launch Finance LLM Demo")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--config_path", default="config/model_config.yaml", 
                       help="Path to model configuration")
    parser.add_argument("--port", type=int, default=7860, help="Port to run demo")
    parser.add_argument("--share", action="store_true", help="Create shareable link")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    # Create demo
    demo_app = FinanceLLMDemo(args.model_path, args.config_path)
    demo = demo_app.create_demo()
    
    # Launch demo
    logger.info(f"Launching demo on port {args.port}")
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )

if __name__ == "__main__":
    main()