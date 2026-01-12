"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing

This script:
1. Creates a stratified sample of 10,000-15,000 complaints
2. Implements text chunking using LangChain's RecursiveCharacterTextSplitter
3. Generates embeddings using sentence-transformers/all-MiniLM-L6-v2
4. Creates and persists a vector store using ChromaDB
5. Stores metadata alongside embeddings for traceability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# ChromaDB for vector store
import chromadb
from chromadb.config import Settings

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Create directories if they don't exist
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Task 2: Text Chunking, Embedding, and Vector Store Indexing")
print("=" * 60)


def load_filtered_data():
    """
    Load the filtered and cleaned dataset from Task 1.
    If filtered_complaints.csv doesn't exist, try to load from raw data.
    """
    filtered_file = PROJECT_ROOT / "data" / "filtered_complaints.csv"
    
    if filtered_file.exists():
        print(f"\n✓ Loading filtered dataset from: {filtered_file}")
        df = pd.read_csv(filtered_file, low_memory=False)
        print(f"✓ Loaded {len(df):,} complaints")
        return df
    else:
        print(f"\n⚠ Filtered dataset not found at {filtered_file}")
        print("Attempting to load from raw data...")
        # Try to load raw data (this would require running Task 1 first)
        raw_file = DATA_RAW / "complaints.csv"
        if raw_file.exists():
            print(f"Loading raw data from: {raw_file}")
            print("Note: This will require filtering. Consider running Task 1 first.")
            # For now, return None - user should run Task 1 first
            return None
        else:
            raise FileNotFoundError(
                "No data file found. Please run Task 1 first to create filtered_complaints.csv"
            )


def create_stratified_sample(df, target_size=12000, random_state=42):
    """
    Create a stratified sample ensuring proportional representation across products.
    
    Args:
        df: DataFrame with filtered complaints
        target_size: Target number of samples (default 12,000)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with stratified sample
    """
    print("\n" + "=" * 60)
    print("Step 1: Creating Stratified Sample")
    print("=" * 60)
    
    # Identify product column
    product_col = None
    for col in ['Product', 'product', 'product_category']:
        if col in df.columns:
            product_col = col
            break
    
    if not product_col:
        raise ValueError("Could not find product column in dataset")
    
    # Get product distribution
    product_counts = df[product_col].value_counts()
    product_proportions = df[product_col].value_counts(normalize=True)
    
    print(f"\nOriginal dataset: {len(df):,} complaints")
    print(f"\nProduct distribution:")
    for product, count in product_counts.items():
        pct = product_proportions[product] * 100
        print(f"  - {product}: {count:,} ({pct:.2f}%)")
    
    # Calculate sample size per product
    sample_sizes = {}
    total_sampled = 0
    
    for product in product_counts.index:
        proportion = product_proportions[product]
        sample_size = max(1, int(target_size * proportion))
        # Don't sample more than available
        sample_size = min(sample_size, product_counts[product])
        sample_sizes[product] = sample_size
        total_sampled += sample_size
    
    # Adjust if total is less than target
    if total_sampled < target_size:
        # Add more samples from largest products
        remaining = target_size - total_sampled
        for product in product_counts.index:
            if remaining <= 0:
                break
            available = product_counts[product] - sample_sizes[product]
            to_add = min(remaining, available)
            sample_sizes[product] += to_add
            remaining -= to_add
            total_sampled += to_add
    
    print(f"\nTarget sample size: {target_size:,}")
    print(f"Actual sample size: {total_sampled:,}")
    print(f"\nSample sizes per product:")
    for product, size in sample_sizes.items():
        print(f"  - {product}: {size:,}")
    
    # Perform stratified sampling
    sampled_dfs = []
    for product, size in sample_sizes.items():
        product_df = df[df[product_col] == product]
        if len(product_df) > size:
            sampled = product_df.sample(n=size, random_state=random_state)
        else:
            sampled = product_df
        sampled_dfs.append(sampled)
    
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n✓ Stratified sample created: {len(sampled_df):,} complaints")
    
    return sampled_df


def chunk_texts(df, chunk_size=500, chunk_overlap=50):
    """
    Chunk the complaint narratives using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        df: DataFrame with complaint narratives
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
    
    Returns:
        List of dictionaries containing chunks and metadata
    """
    print("\n" + "=" * 60)
    print("Step 2: Text Chunking")
    print("=" * 60)
    
    # Identify narrative column
    narrative_col = None
    for col in ['narrative_cleaned', 'Consumer complaint narrative', 
                'consumer_complaint_narrative', 'complaint_narrative']:
        if col in df.columns:
            narrative_col = col
            break
    
    if not narrative_col:
        raise ValueError("Could not find narrative column in dataset")
    
    # Identify other metadata columns
    product_col = None
    for col in ['Product', 'product', 'product_category']:
        if col in df.columns:
            product_col = col
            break
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print(f"\nChunking parameters:")
    print(f"  - Chunk size: {chunk_size} characters")
    print(f"  - Chunk overlap: {chunk_overlap} characters")
    print(f"  - Total narratives to chunk: {len(df):,}")
    
    # Process each complaint
    all_chunks = []
    complaint_id_col = None
    
    # Try to find complaint ID column
    for col in ['Complaint ID', 'complaint_id', 'Complaint ID:', 'id']:
        if col in df.columns:
            complaint_id_col = col
            break
    
    if not complaint_id_col:
        # Create a synthetic ID
        df['complaint_id'] = df.index.astype(str)
        complaint_id_col = 'complaint_id'
    
    print(f"\nProcessing complaints...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        narrative = str(row[narrative_col]) if pd.notna(row[narrative_col]) else ""
        
        if not narrative or len(narrative.strip()) == 0:
            continue
        
        # Split into chunks
        chunks = text_splitter.split_text(narrative)
        
        # Create metadata for each chunk
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_data = {
                'chunk_id': f"{row[complaint_id_col]}_chunk_{chunk_idx}",
                'complaint_id': str(row[complaint_id_col]),
                'chunk_text': chunk_text,
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks),
                'product_category': str(row[product_col]) if product_col else "Unknown",
                'product': str(row.get('Product', 'Unknown')),
                'issue': str(row.get('Issue', 'Unknown')),
                'sub_issue': str(row.get('Sub-issue', 'Unknown')),
                'company': str(row.get('Company', 'Unknown')),
                'state': str(row.get('State', 'Unknown')),
                'date_received': str(row.get('Date received', 'Unknown')),
            }
            all_chunks.append(chunk_data)
    
    print(f"\n✓ Chunking complete!")
    print(f"  - Total chunks created: {len(all_chunks):,}")
    print(f"  - Average chunks per complaint: {len(all_chunks) / len(df):.2f}")
    
    return all_chunks


def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2", batch_size=32):
    """
    Generate embeddings for all text chunks using sentence-transformers.
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Name of the embedding model
        batch_size: Batch size for embedding generation
    
    Returns:
        List of embeddings (numpy arrays)
    """
    print("\n" + "=" * 60)
    print("Step 3: Generating Embeddings")
    print("=" * 60)
    
    print(f"\nLoading embedding model: {model_name}")
    print("This may take a minute on first run (downloading model)...")
    
    model = SentenceTransformer(model_name)
    
    print(f"✓ Model loaded successfully")
    print(f"  - Model dimension: {model.get_sentence_embedding_dimension()}")
    print(f"  - Total chunks to embed: {len(chunks):,}")
    
    # Extract text from chunks
    texts = [chunk['chunk_text'] for chunk in chunks]
    
    print(f"\nGenerating embeddings (batch size: {batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"\n✓ Embeddings generated successfully!")
    print(f"  - Embedding shape: {embeddings.shape}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, model


def create_vector_store(chunks, embeddings, collection_name="complaint_chunks"):
    """
    Create a ChromaDB vector store and store chunks with metadata.
    
    Args:
        chunks: List of chunk dictionaries
        embeddings: Numpy array of embeddings
        collection_name: Name for the ChromaDB collection
    
    Returns:
        ChromaDB collection object
    """
    print("\n" + "=" * 60)
    print("Step 4: Creating Vector Store")
    print("=" * 60)
    
    # Initialize ChromaDB client
    chroma_db_path = VECTOR_STORE_DIR / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    
    print(f"\nChromaDB path: {chroma_db_path}")
    
    # Delete existing collection if it exists (for re-running)
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create or get collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "CFPB complaint chunks with embeddings"}
    )
    
    print(f"✓ Created collection: {collection_name}")
    
    # Prepare data for ChromaDB
    ids = [chunk['chunk_id'] for chunk in chunks]
    documents = [chunk['chunk_text'] for chunk in chunks]
    metadatas = [
        {
            'complaint_id': chunk['complaint_id'],
            'chunk_index': chunk['chunk_index'],
            'total_chunks': chunk['total_chunks'],
            'product_category': chunk['product_category'],
            'product': chunk['product'],
            'issue': chunk['issue'],
            'sub_issue': chunk['sub_issue'],
            'company': chunk['company'],
            'state': chunk['state'],
            'date_received': chunk['date_received'],
        }
        for chunk in chunks
    ]
    
    # Convert embeddings to list format
    embeddings_list = embeddings.tolist()
    
    print(f"\nAdding {len(chunks):,} chunks to vector store...")
    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✓ Vector store created successfully!")
    print(f"  - Total chunks stored: {collection.count()}")
    
    # Save metadata about the vector store
    metadata_file = VECTOR_STORE_DIR / "vector_store_metadata.json"
    metadata_info = {
        'collection_name': collection_name,
        'total_chunks': len(chunks),
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dimension': embeddings.shape[1],
        'chunk_size': 500,
        'chunk_overlap': 50,
        'created_at': datetime.now().isoformat(),
        'sample_size': len(set([c['complaint_id'] for c in chunks]))
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_info, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_file}")
    
    return collection


def main():
    """Main execution function."""
    try:
        # Step 1: Load filtered data
        df = load_filtered_data()
        if df is None:
            print("\n⚠ Please run Task 1 first to create filtered_complaints.csv")
            return
        
        # Step 2: Create stratified sample
        sampled_df = create_stratified_sample(df, target_size=12000, random_state=42)
        
        # Save the sample for reference
        sample_file = DATA_PROCESSED / "task2_sample_complaints.csv"
        sampled_df.to_csv(sample_file, index=False)
        print(f"\n✓ Sample saved to: {sample_file}")
        
        # Step 3: Chunk texts
        chunks = chunk_texts(sampled_df, chunk_size=500, chunk_overlap=50)
        
        # Step 4: Generate embeddings
        embeddings, model = generate_embeddings(chunks, batch_size=32)
        
        # Step 5: Create vector store
        collection = create_vector_store(chunks, embeddings)
        
        # Test the vector store
        print("\n" + "=" * 60)
        print("Testing Vector Store")
        print("=" * 60)
        
        # Sample query
        test_query = "credit card billing dispute"
        query_embedding = model.encode([test_query])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        
        print(f"\nTest query: '{test_query}'")
        print(f"Retrieved {len(results['ids'][0])} results")
        print("\nTop result:")
        print(f"  Chunk ID: {results['ids'][0][0]}")
        print(f"  Product: {results['metadatas'][0][0]['product_category']}")
        print(f"  Text preview: {results['documents'][0][0][:200]}...")
        
        print("\n" + "=" * 60)
        print("Task 2 Complete!")
        print("=" * 60)
        print(f"\n✓ Vector store saved to: {VECTOR_STORE_DIR / 'chroma_db'}")
        print(f"✓ Total chunks indexed: {collection.count():,}")
        print(f"✓ Ready for Task 3 (RAG pipeline)")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

