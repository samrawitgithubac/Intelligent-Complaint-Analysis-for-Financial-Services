"""
Interactive Chat Interface for RAG-Powered Complaint Analysis

This is the main application file that provides a Gradio-based web interface
for querying customer complaints using the RAG pipeline.
"""

import gradio as gr
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline


# Global variable for RAG pipeline
rag_pipeline = None


def initialize_rag():
    """Initialize the RAG pipeline (called once at startup)."""
    global rag_pipeline
    
    if rag_pipeline is None:
        print("Initializing RAG Pipeline...")
        
        # Check for embeddings file
        embeddings_file = None
        possible_paths = [
            "data/processed/complaint_embeddings.parquet",
            "complaint_embeddings.parquet",
            "../data/processed/complaint_embeddings.parquet"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                embeddings_file = path
                print(f"Found embeddings file: {path}")
                break
        
        if embeddings_file is None:
            print("⚠ Warning: Embeddings file not found. Using ChromaDB only.")
        
        # Initialize pipeline
        try:
            rag_pipeline = RAGPipeline(
                vector_store_path="vector_store",
                embeddings_file=embeddings_file,
                top_k=5,
                device="cpu"
            )
            print("✓ RAG Pipeline initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            return False
    
    return True


def format_sources(sources):
    """Format source chunks for display."""
    if not sources:
        return "No sources retrieved."
    
    formatted = []
    for i, source in enumerate(sources[:5], 1):  # Show top 5 sources
        metadata = source.get('metadata', {})
        product = metadata.get('product_category', 'Unknown Product')
        issue = metadata.get('issue', 'Unknown Issue')
        text = source['text']
        
        # Truncate long text
        if len(text) > 300:
            text = text[:300] + "..."
        
        formatted.append(
            f"**Source {i}** (Product: {product}, Issue: {issue})\n"
            f"{text}\n"
            f"{'─' * 60}\n"
        )
    
    return "\n".join(formatted)


def query_complaints(question, history):
    """
    Query the RAG pipeline and return formatted response.
    
    Args:
        question: User's question
        history: Chat history (for Gradio ChatInterface)
    
    Returns:
        Tuple of (answer, sources_text, updated_history)
    """
    global rag_pipeline
    
    if rag_pipeline is None:
        if not initialize_rag():
            return (
                "Error: RAG pipeline could not be initialized. Please check the logs.",
                "No sources available.",
                history
            )
    
    if not question or question.strip() == "":
        return ("Please enter a question.", "", history)
    
    try:
        # Query the RAG pipeline
        result = rag_pipeline.query(question.strip())
        
        answer = result['answer']
        sources = result['sources']
        num_sources = result['num_sources']
        
        # Format sources
        sources_text = format_sources(sources)
        
        # Add source count to answer
        if num_sources > 0:
            answer = f"{answer}\n\n*Based on {num_sources} retrieved complaint(s)*"
        
        # Update history for chat interface
        if history is None:
            history = []
        
        history.append([question, answer])
        
        return answer, sources_text, history
    
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return (error_msg, "No sources available.", history)


def create_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize RAG pipeline
    if not initialize_rag():
        print("⚠ Warning: RAG pipeline initialization failed. The app may not work correctly.")
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin-top: 10px;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=css, title="CrediTrust Complaint Analysis") as demo:
        gr.Markdown(
            """
            # 🏦 CrediTrust Financial - Intelligent Complaint Analysis
            
            Ask questions about customer complaints across Credit Cards, Personal Loans, 
            Savings Accounts, and Money Transfers.
            
            **Example questions:**
            - "Why are people unhappy with Credit Cards?"
            - "What are the main issues with Personal Loans?"
            - "What problems do customers face with Money Transfers?"
            - "What are the most common billing disputes?"
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_copy_button=True
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Ask a question",
                        placeholder="Enter your question about customer complaints...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Conversation", variant="secondary")
            
            with gr.Column(scale=1):
                sources_display = gr.Textbox(
                    label="Retrieved Sources",
                    lines=20,
                    max_lines=25,
                    interactive=False,
                    elem_classes=["source-box"]
                )
        
        # Event handlers
        def submit_question(question, history):
            answer, sources, updated_history = query_complaints(question, history)
            return "", updated_history, sources, updated_history
        
        def clear_conversation():
            return None, "", None
        
        # Connect events
        submit_btn.click(
            fn=submit_question,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot, sources_display, chatbot]
        )
        
        question_input.submit(
            fn=submit_question,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot, sources_display, chatbot]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, sources_display, chatbot]
        )
        
        # Examples
        gr.Examples(
            examples=[
                "Why are people unhappy with Credit Cards?",
                "What are the main issues with Personal Loans?",
                "What problems do customers face with Money Transfers?",
                "What are the most common billing disputes?",
                "What are the top complaints across all products?",
            ],
            inputs=question_input
        )
        
        gr.Markdown(
            """
            ---
            **Note:** This tool uses RAG (Retrieval-Augmented Generation) to answer questions 
            based on real customer complaint data. Sources are displayed below for verification.
            """
        )
    
    return demo


def main():
    """Main function to launch the app."""
    print("=" * 80)
    print("CrediTrust Financial - Intelligent Complaint Analysis")
    print("=" * 80)
    print("\nStarting Gradio interface...")
    print("The app will be available at http://localhost:7860")
    print("\nPress Ctrl+C to stop the server.\n")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()

