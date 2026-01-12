# Intelligent Complaint Analysis for Financial Services

Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights

## 📋 Project Overview

This project develops an intelligent complaint-answering chatbot that empowers product, support, and compliance teams to understand customer pain points across financial services. The system uses **Retrieval-Augmented Generation (RAG)** to transform unstructured complaint data into strategic insights.

### Business Objective

CrediTrust Financial is a fast-growing digital finance company serving East African markets. With over 500,000 users and operations in three countries, they receive thousands of customer complaints per month. This tool helps internal stakeholders like Product Managers quickly identify trends and get evidence-backed answers in seconds.

### Key Performance Indicators (KPIs)

1. **Decrease time to identify trends**: From days to minutes
2. **Empower non-technical teams**: Get answers without data analysts
3. **Proactive problem identification**: Shift from reactive to proactive based on real-time feedback

## 🎯 Features

- **Exploratory Data Analysis**: Comprehensive analysis of CFPB complaint data
- **Text Preprocessing**: Cleaning and normalization of complaint narratives
- **Stratified Sampling**: Proportional representation across product categories
- **Text Chunking**: Intelligent splitting of long narratives for better embeddings
- **Vector Embeddings**: Semantic search using sentence-transformers
- **Vector Store**: ChromaDB-based storage with metadata
- **RAG Pipeline**: (Coming in Task 3) Retrieval and generation pipeline
- **Interactive UI**: (Coming in Task 4) User-friendly chat interface

## 📁 Project Structure

```
Intelligent-Complaint-Analysis-for-Financial-Services/
├── .gitignore                          # Git ignore rules
├── README.md                           # This file
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── raw/                           # Raw CFPB complaint dataset
│   │   └── complaints.csv
│   └── processed/                    # Processed and filtered data
│       ├── filtered_complaints.csv    # Task 1 output
│       └── task2_sample_complaints.csv # Task 2 sample
│
├── notebooks/
│   ├── __init__.py
│   ├── README.md                      # Notebooks documentation
│   └── task1_eda_preprocessing.ipynb # Task 1: EDA and preprocessing
│
├── src/
│   ├── __init__.py
│   └── task2_chunking_embedding.py   # Task 2: Chunking and embedding
│
├── vector_store/                      # Vector database storage
│   └── chroma_db/                    # ChromaDB persistent storage
│
└── tests/
    └── __init__.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/samrawitgithubac/Intelligent-Complaint-Analysis-for-Financial-Services.git
   cd Intelligent-Complaint-Analysis-for-Financial-Services
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the CFPB dataset**
   - Download the CFPB complaint dataset from: https://www.consumerfinance.gov/data-research/consumer-complaints/
   - Place the CSV file in `data/raw/complaints.csv`

## 📚 Tasks Overview

### Task 1: Exploratory Data Analysis and Data Preprocessing ✅

**Objective**: Understand the structure, content, and quality of complaint data and prepare it for the RAG pipeline.

**Deliverables**:
- Jupyter notebook with comprehensive EDA
- Cleaned and filtered dataset (`data/filtered_complaints.csv`)
- Analysis of product distribution and narrative characteristics

**How to run**:
```bash
# Open the Jupyter notebook
jupyter notebook notebooks/task1_eda_preprocessing.ipynb

# Or run as a script (if converted)
python notebooks/task1_eda_preprocessing.py
```

**Key Features**:
- Product distribution analysis
- Narrative length analysis (word count)
- Text cleaning and normalization
- Filtering for target products (Credit Card, Personal Loan, Savings Account, Money Transfers)

### Task 2: Text Chunking, Embedding, and Vector Store Indexing ✅

**Objective**: Convert cleaned text narratives into a format suitable for efficient semantic search.

**Deliverables**:
- Stratified sample of 10,000-15,000 complaints
- Text chunks with embeddings
- ChromaDB vector store with metadata

**How to run**:
```bash
python src/task2_chunking_embedding.py
```

**Key Features**:
- **Stratified Sampling**: Proportional representation across products
- **Text Chunking**: 500 characters with 50 character overlap
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Store**: ChromaDB with complete metadata

**Parameters**:
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Sample size: ~12,000 complaints
- Embedding dimension: 384

### Task 3: Building the RAG Core Logic and Evaluation 🚧

**Objective**: Build the retrieval and generation pipeline using the pre-built vector store.

**Status**: Coming soon

**Planned Features**:
- Question embedding and similarity search
- Prompt engineering for LLM
- Response generation with context
- Evaluation framework

### Task 4: Creating an Interactive Chat Interface 🚧

**Objective**: Build a user-friendly interface for non-technical users.

**Status**: Coming soon

**Planned Features**:
- Gradio/Streamlit web interface
- Source citation display
- Response streaming
- Multi-product querying

## 🔧 Configuration

### Data Requirements

- **Raw Data**: CFPB complaint dataset (5.63 GB)
- **Filtered Data**: Created by Task 1
- **Sample Data**: Created by Task 2 (10K-15K complaints)

### Vector Store

The vector store is stored in `vector_store/chroma_db/` and includes:
- Text chunks with embeddings
- Metadata (complaint_id, product_category, issue, etc.)
- Collection name: `complaint_chunks`

## 📊 Data Products

### Target Products

The system focuses on four main financial products:
1. **Credit Card**
2. **Personal Loan**
3. **Savings Account**
4. **Money Transfers**

### Data Sources

- **CFPB Dataset**: Consumer Financial Protection Bureau complaint database
- Contains real customer complaints with narratives, product information, and metadata

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **LangChain**: Text chunking and RAG framework
- **Sentence Transformers**: Embedding generation
- **ChromaDB**: Vector database
- **Jupyter Notebooks**: Interactive development

## 📝 Key Decisions

### Chunking Strategy
- **Chunk Size**: 500 characters
  - Balances context preservation with embedding quality
  - Prevents information loss in very long narratives
- **Overlap**: 50 characters
  - Ensures continuity between chunks
  - Prevents splitting important context

### Embedding Model
- **Model**: `all-MiniLM-L6-v2`
  - Fast and efficient (384 dimensions)
  - Good balance between quality and speed
  - Widely used in production RAG systems

### Sampling Strategy
- **Stratified Sampling**: Ensures proportional representation
- **Sample Size**: 10,000-15,000 complaints
  - Sufficient for learning chunking and embedding
  - Manageable for standard hardware

## 🤝 Contributing

This is a project for the Intelligent Complaint Analysis challenge. For contributions:

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📅 Timeline

- **Challenge Introduction**: December 31, 2025
- **Interim Submission**: January 4, 2026 (Tasks 1-2)
- **Final Submission**: January 13, 2026 (All tasks)

## 📄 License

This project is part of an educational challenge. Please refer to the challenge guidelines for usage terms.

## 👥 Team

**Facilitators**:
- Kerod
- Mahbubah
- Filimon
- Smegnsh

## 🔗 Resources

### Documentation
- [Gradio Documentation](https://www.gradio.app/docs)
- [Streamlit Documentation](https://docs.streamlit.io/library/api-reference/chat)
- [ChromaDB Getting Started](https://docs.trychroma.com/getting-started)
- [FAISS Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

### Tutorials
- [RAG with Hugging Face](https://huggingface.co/blog/rag)
- [LangChain RAG Examples](https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb)

### Data Sources
- [CFPB Consumer Complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/)

## 🐛 Troubleshooting

### Common Issues

1. **Memory errors when loading data**
   - The dataset is large (5.63 GB). Use the chunked loading approach in Task 1.
   - Ensure you have sufficient RAM (recommended: 8GB+)

2. **Vector store not found**
   - Run Task 2 first to create the vector store
   - Check that `vector_store/chroma_db/` exists

3. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.8+ is installed

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Status**: Tasks 1-2 Complete ✅ | Tasks 3-4 In Progress 🚧
