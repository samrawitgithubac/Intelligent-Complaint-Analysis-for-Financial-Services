"""
Evaluation script for RAG Pipeline

This module provides evaluation functionality to test the RAG system
with representative questions and analyze the quality of responses.
"""

import pandas as pd
from typing import List, Dict
from rag_pipeline import RAGPipeline


class RAGEvaluator:
    """
    Evaluator for RAG pipeline quality assessment.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the evaluator.
        
        Args:
            rag_pipeline: Initialized RAGPipeline instance
        """
        self.rag_pipeline = rag_pipeline
    
    def get_test_questions(self) -> List[Dict[str, str]]:
        """
        Get a list of representative test questions.
        
        Returns:
            List of dictionaries with 'question' and 'category' keys
        """
        questions = [
            {
                'question': 'Why are people unhappy with Credit Cards?',
                'category': 'Product-specific'
            },
            {
                'question': 'What are the main issues with Personal Loans?',
                'category': 'Product-specific'
            },
            {
                'question': 'What problems do customers face with Money Transfers?',
                'category': 'Product-specific'
            },
            {
                'question': 'What are the most common billing disputes?',
                'category': 'Issue-specific'
            },
            {
                'question': 'How do customers complain about account management?',
                'category': 'Issue-specific'
            },
            {
                'question': 'What are the top complaints across all products?',
                'category': 'Cross-product'
            },
            {
                'question': 'Compare issues between Credit Cards and Personal Loans',
                'category': 'Comparison'
            },
            {
                'question': 'What fraud-related complaints have been reported?',
                'category': 'Compliance'
            },
            {
                'question': 'What are customers saying about interest rates?',
                'category': 'Feature-specific'
            },
            {
                'question': 'What issues are customers reporting with Savings Accounts?',
                'category': 'Product-specific'
            }
        ]
        return questions
    
    def evaluate_question(self, question: str) -> Dict:
        """
        Evaluate a single question.
        
        Args:
            question: Question to evaluate
        
        Returns:
            Dictionary with evaluation results
        """
        result = self.rag_pipeline.query(question)
        
        # Extract source information
        sources_summary = []
        for i, source in enumerate(result['sources'][:2], 1):  # Show top 2 sources
            metadata = source.get('metadata', {})
            product = metadata.get('product_category', 'Unknown')
            issue = metadata.get('issue', 'Unknown')
            text_preview = source['text'][:150] + "..." if len(source['text']) > 150 else source['text']
            sources_summary.append({
                'source_num': i,
                'product': product,
                'issue': issue,
                'text_preview': text_preview
            })
        
        return {
            'question': question,
            'answer': result['answer'],
            'num_sources': result['num_sources'],
            'sources': sources_summary,
            'quality_score': None,  # To be filled manually
            'comments': None  # To be filled manually
        }
    
    def run_evaluation(self) -> pd.DataFrame:
        """
        Run evaluation on all test questions.
        
        Returns:
            DataFrame with evaluation results
        """
        questions = self.get_test_questions()
        results = []
        
        print("Running RAG Pipeline Evaluation...")
        print("=" * 80)
        
        for i, q_dict in enumerate(questions, 1):
            question = q_dict['question']
            category = q_dict['category']
            
            print(f"\n[{i}/{len(questions)}] Evaluating: {question}")
            print(f"Category: {category}")
            print("-" * 80)
            
            eval_result = self.evaluate_question(question)
            eval_result['category'] = category
            results.append(eval_result)
            
            print(f"Answer: {eval_result['answer'][:200]}...")
            print(f"Retrieved {eval_result['num_sources']} sources")
        
        print("\n" + "=" * 80)
        print("Evaluation complete!")
        
        return pd.DataFrame(results)
    
    def generate_evaluation_table(self, results_df: pd.DataFrame) -> str:
        """
        Generate a markdown table for the evaluation results.
        
        Args:
            results_df: DataFrame with evaluation results
        
        Returns:
            Markdown-formatted table string
        """
        markdown_lines = []
        markdown_lines.append("## RAG Pipeline Evaluation Results\n")
        markdown_lines.append("| Question | Generated Answer | Retrieved Sources | Quality Score | Comments/Analysis |")
        markdown_lines.append("|----------|------------------|-------------------|---------------|-------------------|")
        
        for _, row in results_df.iterrows():
            question = row['question']
            answer = row['answer'].replace('\n', ' ')[:200] + "..." if len(row['answer']) > 200 else row['answer'].replace('\n', ' ')
            
            # Format sources
            sources_text = ""
            for source in row['sources']:
                sources_text += f"**Source {source['source_num']}**: {source['product']} - {source['issue']}<br>"
                sources_text += f"*{source['text_preview']}*<br><br>"
            
            quality_score = row.get('quality_score', 'TBD')
            comments = row.get('comments', 'TBD')
            
            markdown_lines.append(
                f"| {question} | {answer} | {sources_text} | {quality_score} | {comments} |"
            )
        
        return "\n".join(markdown_lines)


def main():
    """Main function to run evaluation."""
    # Initialize RAG pipeline
    # Note: Update paths based on your setup
    embeddings_file = "data/processed/complaint_embeddings.parquet"  # Update if different
    vector_store_path = "vector_store"
    
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline(
        vector_store_path=vector_store_path,
        embeddings_file=embeddings_file,
        top_k=5
    )
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag)
    
    # Run evaluation
    results_df = evaluator.run_evaluation()
    
    # Save results
    results_df.to_csv("evaluation_results.csv", index=False)
    print(f"\n✓ Evaluation results saved to evaluation_results.csv")
    
    # Generate markdown table
    markdown_table = evaluator.generate_evaluation_table(results_df)
    
    # Save markdown
    with open("evaluation_table.md", "w", encoding="utf-8") as f:
        f.write(markdown_table)
    
    print(f"✓ Evaluation table saved to evaluation_table.md")
    
    # Display summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total questions evaluated: {len(results_df)}")
    print(f"Average sources retrieved: {results_df['num_sources'].mean():.2f}")
    print(f"Questions with sources: {(results_df['num_sources'] > 0).sum()}/{len(results_df)}")
    
    return results_df


if __name__ == "__main__":
    results = main()

