"""
Task 3: RAG Pipeline Evaluation Runner

This script runs the evaluation for Task 3 and generates the evaluation report.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate_rag import RAGEvaluator
from src.rag_pipeline import RAGPipeline


def main():
    """Run Task 3 evaluation."""
    print("=" * 80)
    print("Task 3: RAG Pipeline Evaluation")
    print("=" * 80)
    print()
    
    # Initialize RAG pipeline
    print("Step 1: Initializing RAG Pipeline...")
    embeddings_file = None
    
    # Check for embeddings file in common locations
    possible_paths = [
        "data/processed/complaint_embeddings.parquet",
        "complaint_embeddings.parquet",
        "../data/processed/complaint_embeddings.parquet"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            embeddings_file = path
            print(f"  ✓ Found embeddings file: {path}")
            break
    
    if embeddings_file is None:
        print("  ⚠ Warning: Embeddings file not found. Will use ChromaDB only.")
        print("  Make sure you have either:")
        print("    - complaint_embeddings.parquet in data/processed/")
        print("    - A ChromaDB vector store in vector_store/")
    
    try:
        rag = RAGPipeline(
            vector_store_path="vector_store",
            embeddings_file=embeddings_file,
            top_k=5
        )
        print("  ✓ RAG Pipeline initialized successfully!")
    except Exception as e:
        print(f"  ✗ Error initializing RAG pipeline: {e}")
        print("  Please ensure the vector store or embeddings file is available.")
        return
    
    print()
    print("Step 2: Running Evaluation...")
    print()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag)
    
    # Run evaluation
    try:
        results_df = evaluator.run_evaluation()
        
        # Save results
        output_dir = Path("evaluation_output")
        output_dir.mkdir(exist_ok=True)
        
        results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
        print(f"\n  ✓ Evaluation results saved to {output_dir / 'evaluation_results.csv'}")
        
        # Generate markdown table
        markdown_table = evaluator.generate_evaluation_table(results_df)
        
        with open(output_dir / "evaluation_table.md", "w", encoding="utf-8") as f:
            f.write(markdown_table)
        
        print(f"  ✓ Evaluation table saved to {output_dir / 'evaluation_table.md'}")
        
        # Display summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total questions evaluated: {len(results_df)}")
        print(f"Average sources retrieved: {results_df['num_sources'].mean():.2f}")
        print(f"Questions with sources: {(results_df['num_sources'] > 0).sum()}/{len(results_df)}")
        print()
        print("Next steps:")
        print("1. Review the evaluation_results.csv file")
        print("2. Manually add quality scores (1-5) and comments to the CSV")
        print("3. Re-run this script or manually update evaluation_table.md")
        print("4. Include the evaluation table in your final report")
        
    except Exception as e:
        print(f"\n  ✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

