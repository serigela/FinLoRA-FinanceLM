#!/usr/bin/env python3
"""
Simple demo for Finance LLM LoRA project (without model loading)
"""

import gradio as gr
from typing import Dict, Any
import random

class FinanceLLMSimpleDemo:
    """Simple demo interface for Finance LLM without actual model"""
    
    def __init__(self):
        # Sample responses for demonstration
        self.sentiment_responses = {
            "positive": ["positive", "bullish", "optimistic"],
            "negative": ["negative", "bearish", "pessimistic"],
            "neutral": ["neutral", "stable", "unchanged"]
        }
        
        self.headline_responses = {
            "bullish": ["bullish", "positive", "upward"],
            "bearish": ["bearish", "negative", "downward"],
            "neutral": ["neutral", "stable", "sideways"]
        }
    
    def sentiment_analysis(self, text: str) -> str:
        """Simulate sentiment analysis"""
        if not text.strip():
            return "Please enter some text to analyze."
        
        # Simple keyword-based simulation
        positive_keywords = ["strong", "growth", "increase", "beat", "exceed", "good", "profit", "revenue", "up"]
        negative_keywords = ["weak", "decline", "decrease", "miss", "disappoint", "bad", "loss", "down", "fall"]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_count > negative_count:
            return random.choice(self.sentiment_responses["positive"])
        elif negative_count > positive_count:
            return random.choice(self.sentiment_responses["negative"])
        else:
            return random.choice(self.sentiment_responses["neutral"])
    
    def headline_classification(self, text: str) -> str:
        """Simulate headline classification"""
        if not text.strip():
            return "Please enter a headline to classify."
        
        # Simple keyword-based simulation
        bullish_keywords = ["surge", "gains", "soar", "rally", "breakthrough", "beats", "up", "growth", "strong"]
        bearish_keywords = ["plummet", "fall", "decline", "crash", "concerns", "headwinds", "down", "weak", "miss"]
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
        
        if bullish_count > bearish_count:
            return random.choice(self.headline_responses["bullish"])
        elif bearish_count > bullish_count:
            return random.choice(self.headline_responses["bearish"])
        else:
            return random.choice(self.headline_responses["neutral"])
    
    def summarization(self, text: str) -> str:
        """Simulate summarization"""
        if not text.strip():
            return "Please enter text to summarize."
        
        # Simple bullet point generation
        sentences = text.split('.')
        key_points = []
        
        for sentence in sentences[:3]:  # Take first 3 sentences
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:
                key_points.append(f"• {sentence}")
        
        if not key_points:
            key_points = ["• Revenue and performance metrics discussed", "• Market conditions and outlook addressed", "• Strategic initiatives highlighted"]
        
        return "\n".join(key_points)
    
    def financial_qa(self, context: str, question: str) -> str:
        """Simulate financial Q&A"""
        if not context.strip() or not question.strip():
            return "Please provide both context and question."
        
        # Simple keyword matching
        question_lower = question.lower()
        context_lower = context.lower()
        
        if "revenue" in question_lower:
            # Look for revenue numbers in context
            words = context.split()
            for i, word in enumerate(words):
                if "$" in word or "billion" in word or "million" in word:
                    return f"{word} {words[i+1] if i+1 < len(words) else ''}".strip()
            return "Revenue information not clearly specified in context."
        
        elif "what" in question_lower and "when" not in question_lower:
            # Extract key information
            sentences = context.split('.')
            if sentences:
                return sentences[0].strip()
            return "Information not clearly specified in context."
        
        else:
            return "Based on the context provided, the answer relates to the financial information mentioned."
    
    def stock_prediction(self, text: str) -> str:
        """Simulate stock movement prediction"""
        if not text.strip():
            return "Please enter stock-related news."
        
        # Simple keyword-based prediction
        positive_keywords = ["beats", "exceeds", "breakthrough", "strong", "growth", "up", "positive", "good"]
        negative_keywords = ["misses", "disappoints", "concerns", "weak", "decline", "down", "negative", "bad"]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_count > negative_count:
            return f"+{random.uniform(0.5, 3.5):.2f}%"
        elif negative_count > positive_count:
            return f"-{random.uniform(0.5, 2.5):.2f}%"
        else:
            return f"{random.uniform(-0.5, 0.5):.2f}%"
    
    def create_demo(self) -> gr.Blocks:
        """Create Gradio demo interface"""
        
        with gr.Blocks(title="Finance LLM Demo", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🏦 Finance LLM with LoRA Demo")
            gr.Markdown("This demo showcases a multi-task finance LLM fine-tuned with LoRA adapters.")
            gr.Markdown("**Note: This is a simplified demonstration. The actual model would provide more sophisticated responses.**")
            
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
            gr.Markdown("*This demo uses simulated responses to demonstrate the interface. A trained model would provide more sophisticated and accurate results.*")
        
        return demo

def main():
    # Create demo
    demo_app = FinanceLLMSimpleDemo()
    demo = demo_app.create_demo()
    
    # Launch demo
    print("Launching Finance LLM Demo...")
    print("Demo will be available at: http://localhost:7860")
    
    demo.launch(
        server_port=7860,
        share=False,
        server_name="0.0.0.0"
    )

if __name__ == "__main__":
    main()