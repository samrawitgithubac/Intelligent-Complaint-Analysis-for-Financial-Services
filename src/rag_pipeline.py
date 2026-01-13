"""
RAG Pipeline for Intelligent Complaint Analysis

This module implements the core RAG (Retrieval-Augmented Generation) pipeline
for querying customer complaints using semantic search and LLM generation.
"""

import os
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class RAGPipeline:
    """
    RAG Pipeline that combines semantic search with LLM generation.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: Optional[str] = None,
        vector_store_path: str = "vector_store",
        embeddings_file: Optional[str] = None,
        top_k: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the LLM model (if None, uses a default)
            vector_store_path: Path to the ChromaDB vector store
            embeddings_file: Path to pre-built embeddings parquet file
            top_k: Number of chunks to retrieve
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.top_k = top_k
        self.device = device
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize vector store
        self.vector_store_path = vector_store_path
        self.embeddings_file = embeddings_file
        self.vector_store = None
        self.collection = None
        
        # Initialize LLM
        self.llm = None
        self.tokenizer = None
        self._initialize_llm()
        
        # Load vector store
        self._load_vector_store()
    
    def _initialize_llm(self):
        """Initialize the LLM for text generation."""
        # For this project, we'll use a template-based approach with optional LLM
        # This allows the system to work even without GPU resources
        # In production, you would use a proper LLM API or local model
        self.use_pipeline = False
        self.llm = None
        
        # Optionally try to load a lightweight LLM
        if self.llm_model_name:
            try:
                print(f"Attempting to load LLM: {self.llm_model_name}")
                from transformers import pipeline as hf_pipeline
                self.llm = hf_pipeline(
                    "text-generation",
                    model=self.llm_model_name,
                    device=-1 if self.device == "cpu" else 0,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7
                )
                self.use_pipeline = True
                print("✓ LLM loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load LLM model: {e}")
                print("  Will use template-based generation instead")
                self.use_pipeline = False
    
    def _load_vector_store(self):
        """Load the vector store from ChromaDB or embeddings file."""
        # Check if we have a pre-built embeddings file
        if self.embeddings_file and os.path.exists(self.embeddings_file):
            print(f"Loading embeddings from: {self.embeddings_file}")
            self._load_embeddings_from_file()
        else:
            # Try to load from ChromaDB
            print(f"Loading vector store from: {self.vector_store_path}")
            self._load_chromadb()
    
    def _load_embeddings_from_file(self):
        """Load embeddings from a parquet file and create ChromaDB collection."""
        try:
            df = pd.read_parquet(self.embeddings_file)
            print(f"Loaded {len(df):,} chunks from embeddings file")
            print(f"Columns: {list(df.columns)}")
            
            # Initialize ChromaDB
            client = chromadb.PersistentClient(
                path=self.vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            collection_name = "complaint_embeddings"
            try:
                self.collection = client.get_collection(name=collection_name)
                count = self.collection.count()
                if count > 0:
                    print(f"✓ Using existing collection: {collection_name} ({count:,} chunks)")
                else:
                    raise Exception("Collection is empty")
            except:
                self.collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Created new collection: {collection_name}")
                
                # Add embeddings to collection
                print("Adding embeddings to ChromaDB...")
                batch_size = 1000
                
                # Determine column names - handle different possible formats
                text_col = None
                embedding_col = None
                
                # Try common column names
                for col in df.columns:
                    if col.lower() in ['text', 'chunk_text', 'narrative', 'complaint_text']:
                        text_col = col
                    elif col.lower() in ['embedding', 'embeddings', 'vector', 'vectors']:
                        embedding_col = col
                
                # Fallback to first column if not found
                if text_col is None:
                    text_col = df.columns[0]
                
                print(f"Using text column: {text_col}")
                print(f"Using embedding column: {embedding_col if embedding_col else 'Will generate embeddings'}")
                
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    # Generate IDs
                    if 'complaint_id' in batch.columns and 'chunk_index' in batch.columns:
                        ids = [f"{row['complaint_id']}_chunk_{int(row['chunk_index'])}" 
                               for _, row in batch.iterrows()]
                    else:
                        ids = [f"chunk_{j}" for j in range(i, min(i+batch_size, len(df)))]
                    
                    # Get texts
                    texts = batch[text_col].astype(str).tolist()
                    
                    # Get embeddings
                    if embedding_col and embedding_col in batch.columns:
                        # Handle different embedding formats
                        embeddings_raw = batch[embedding_col].tolist()
                        embeddings = []
                        for emb in embeddings_raw:
                            if isinstance(emb, (list, np.ndarray)):
                                embeddings.append(list(emb))
                            elif isinstance(emb, pd.Series):
                                embeddings.append(emb.tolist())
                            else:
                                # Try to convert
                                embeddings.append(list(emb))
                    else:
                        # Generate embeddings if not present
                        print(f"  Generating embeddings for batch {i//batch_size + 1}...")
                        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
                    
                    # Prepare metadata - exclude embedding and text columns
                    metadata_cols = [col for col in batch.columns 
                                    if col not in [text_col, embedding_col] and col != 'embedding']
                    metadatas = []
                    for _, row in batch.iterrows():
                        metadata = {}
                        for col in metadata_cols:
                            val = row[col]
                            # Convert non-serializable types
                            if pd.isna(val):
                                metadata[col] = None
                            elif isinstance(val, (pd.Timestamp, pd.DatetimeTZDtype)):
                                metadata[col] = str(val)
                            elif isinstance(val, (list, np.ndarray)):
                                metadata[col] = str(val)
                            else:
                                metadata[col] = val
                        metadatas.append(metadata)
                    
                    # Add to collection
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )
                    
                    if (i // batch_size + 1) % 10 == 0:
                        print(f"  Processed {min(i+batch_size, len(df)):,}/{len(df):,} chunks")
                
                print(f"✓ Embeddings loaded into ChromaDB ({self.collection.count():,} chunks)")
            
            self.vector_store = client
            
        except Exception as e:
            print(f"Error loading embeddings file: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to ChromaDB loading...")
            self._load_chromadb()
    
    def _load_chromadb(self):
        """Load existing ChromaDB vector store."""
        try:
            client = chromadb.PersistentClient(
                path=self.vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            collection_name = "complaint_embeddings"
            try:
                self.collection = client.get_collection(name=collection_name)
                print(f"✓ Loaded ChromaDB collection: {collection_name}")
                print(f"  Collection contains {self.collection.count()} chunks")
            except:
                print(f"⚠ Collection '{collection_name}' not found in {self.vector_store_path}")
                print("  Please ensure the vector store has been created (Task 2)")
                self.collection = None
            
            self.vector_store = client
            
        except Exception as e:
            print(f"Error loading ChromaDB: {e}")
            self.vector_store = None
            self.collection = None
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a given query.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (overrides self.top_k)
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if self.collection is None:
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        # Embed the query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search the vector store
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        retrieved_chunks = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                chunk = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                }
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def _create_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Create a prompt for the LLM using the query and retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks with text and metadata
        
        Returns:
            Formatted prompt string
        """
        # Combine context chunks
        context_texts = []
        for i, chunk in enumerate(context_chunks, 1):
            chunk_text = chunk['text']
            metadata = chunk.get('metadata', {})
            
            # Add metadata context if available
            product = metadata.get('product_category', 'Unknown')
            issue = metadata.get('issue', 'Unknown')
            
            context_texts.append(
                f"[Source {i}] (Product: {product}, Issue: {issue})\n{chunk_text}\n"
            )
        
        context = "\n".join(context_texts)
        
        # Create prompt
        prompt = f"""You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the retrieved complaint excerpts provided below.

Instructions:
- Use ONLY the information provided in the context to answer the question
- Be concise and specific in your response
- If the context doesn't contain enough information to answer the question, state that clearly
- Focus on actionable insights that would help product managers and support teams
- Mention specific product categories or issues when relevant

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate an answer using the LLM or template-based approach.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks
        
        Returns:
            Generated answer string
        """
        # Create prompt
        prompt = self._create_prompt(query, context_chunks)
        
        # Try to use LLM if available
        if self.use_pipeline and self.llm is not None:
            try:
                # Generate response
                response = self.llm(
                    prompt,
                    max_new_tokens=250,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm.tokenizer.eos_token_id if hasattr(self.llm, 'tokenizer') else None
                )
                
                # Extract generated text
                if isinstance(response, list) and len(response) > 0:
                    generated_text = response[0].get('generated_text', '')
                    # Remove the prompt from the response
                    if generated_text.startswith(prompt):
                        answer = generated_text[len(prompt):].strip()
                    else:
                        answer = generated_text.strip()
                    
                    # Clean up the answer
                    if answer:
                        # Remove any remaining prompt artifacts
                        answer = answer.split("Question:")[0].strip()
                        answer = answer.split("Context:")[0].strip()
                        return answer
                    else:
                        return self._fallback_generate(query, context_chunks)
                else:
                    return self._fallback_generate(query, context_chunks)
            except Exception as e:
                print(f"Error during LLM generation: {e}")
                return self._fallback_generate(query, context_chunks)
        else:
            # Use template-based response (works well for RAG)
            return self._fallback_generate(query, context_chunks)
    
    def _fallback_generate(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Template-based generation method that creates intelligent summaries.
        This works well for RAG and doesn't require a large LLM.
        """
        if not context_chunks:
            return "I couldn't find any relevant complaints to answer your question. Please try rephrasing or asking about a different topic."
        
        # Extract key information from chunks
        products = {}
        issues = {}
        sub_issues = {}
        key_points = []
        companies = {}
        
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            product = metadata.get('product_category', 'Unknown Product')
            issue = metadata.get('issue', 'Unknown Issue')
            sub_issue = metadata.get('sub_issue', '')
            company = metadata.get('company', '')
            text = chunk['text']
            
            products[product] = products.get(product, 0) + 1
            issues[issue] = issues.get(issue, 0) + 1
            if sub_issue:
                sub_issues[sub_issue] = sub_issues.get(sub_issue, 0) + 1
            if company:
                companies[company] = companies.get(company, 0) + 1
            
            # Extract meaningful excerpt (first 250 chars or up to sentence end)
            excerpt = text[:250]
            if len(text) > 250:
                # Try to end at a sentence
                last_period = excerpt.rfind('.')
                if last_period > 150:
                    excerpt = excerpt[:last_period + 1]
                else:
                    excerpt += "..."
            key_points.append(excerpt)
        
        # Build intelligent response based on query type
        response_parts = []
        
        # Analyze query to determine response style
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['why', 'unhappy', 'problem', 'issue']):
            # Problem-focused response
            response_parts.append(f"Based on analysis of {len(context_chunks)} relevant complaint(s), here are the key issues:\n")
            
            if issues:
                top_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]
                response_parts.append("\n**Primary Issues:**")
                for issue_name, count in top_issues:
                    response_parts.append(f"• {issue_name} ({count} complaint{'s' if count > 1 else ''})")
            
            if sub_issues:
                top_sub_issues = sorted(sub_issues.items(), key=lambda x: x[1], reverse=True)[:3]
                if top_sub_issues:
                    response_parts.append("\n**Specific Sub-Issues:**")
                    for sub_issue_name, count in top_sub_issues:
                        response_parts.append(f"• {sub_issue_name}")
        
        elif any(word in query_lower for word in ['what', 'main', 'common', 'top']):
            # Summary-focused response
            response_parts.append(f"Here's a summary based on {len(context_chunks)} relevant complaint(s):\n")
            
            if products:
                response_parts.append(f"\n**Affected Products:** {', '.join(sorted(products.keys()))}")
            
            if issues:
                top_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]
                response_parts.append(f"\n**Most Common Issues:**")
                for issue_name, count in top_issues:
                    response_parts.append(f"• {issue_name}")
        
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            # Comparison-focused response
            response_parts.append(f"Comparison based on {len(context_chunks)} relevant complaint(s):\n")
            
            if products:
                response_parts.append(f"\n**Products Mentioned:** {', '.join(sorted(products.keys()))}")
                for product_name, count in sorted(products.items(), key=lambda x: x[1], reverse=True):
                    product_issues = [chunk.get('metadata', {}).get('issue', 'Unknown') 
                                     for chunk in context_chunks 
                                     if chunk.get('metadata', {}).get('product_category') == product_name]
                    if product_issues:
                        top_product_issue = max(set(product_issues), key=product_issues.count)
                        response_parts.append(f"• **{product_name}**: {count} complaint(s), main issue: {top_product_issue}")
        
        else:
            # General response
            response_parts.append(f"Based on {len(context_chunks)} relevant complaint(s):\n")
            
            if products:
                response_parts.append(f"\n**Products:** {', '.join(sorted(products.keys()))}")
            
            if issues:
                top_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)[:3]
                response_parts.append(f"\n**Key Issues:** {', '.join([issue for issue, _ in top_issues])}")
        
        # Add key excerpts
        response_parts.append("\n\n**Key Complaint Excerpts:**")
        for i, point in enumerate(key_points[:3], 1):
            response_parts.append(f"\n{i}. {point}")
        
        return "\n".join(response_parts)
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find any relevant complaints to answer your question. Please try rephrasing or asking about a different topic.",
                'sources': [],
                'num_sources': 0
            }
        
        # Generate answer
        answer = self.generate(question, retrieved_chunks)
        
        return {
            'answer': answer,
            'sources': retrieved_chunks,
            'num_sources': len(retrieved_chunks)
        }

