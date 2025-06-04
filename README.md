# RAG Nutrition Textbook System

A Retrieval-Augmented Generation (RAG) system that processes nutrition textbooks to provide accurate, context-aware answers to nutrition-related questions using local language models.

## Overview

This system combines document processing, semantic search, and local language model inference to create an intelligent question-answering system for nutrition education. It downloads and processes a nutrition textbook, creates embeddings for semantic search, and uses a local Gemma model to generate contextually relevant answers.

## Features

- **Automatic PDF Processing**: Downloads and extracts text from nutrition textbooks
- **Intelligent Text Chunking**: Splits documents into semantically meaningful chunks
- **Semantic Search**: Uses sentence transformers for finding relevant context
- **Local LLM Integration**: Runs Gemma models locally with GPU acceleration
- **Memory Optimization**: Supports 4-bit quantization for lower memory usage
- **Interactive Q&A**: Provides detailed, contextual answers to nutrition questions

## Requirements

### Hardware
- **GPU Memory Requirements**:
  - Minimum 5GB: Gemma 2B with 4-bit quantization
  - 8GB+: Gemma 2B in float16
  - 19GB+: Gemma 7B in float16
- CUDA-compatible GPU (recommended for performance)

### Software Dependencies

```bash
# Core dependencies
pip install torch>=2.1.1
pip install transformers==4.40.1
pip install sentence-transformers==2.6.1
pip install accelerate
pip install bitsandbytes

# Document processing
pip install PyMuPDF
pip install pandas
pip install spacy

# Additional utilities
pip install tqdm
pip install matplotlib
pip install requests

# Optional: Flash Attention (requires NVIDIA GPU compute capability 8.0+)
pip install flash-attn --no-build-isolation
```

## Installation

1. **Clone or download the script**
2. **Install dependencies** (see requirements above)
3. **Download spaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Setup

The system automatically handles most setup tasks:

```python
# The script will automatically:
# 1. Download the nutrition textbook PDF
# 2. Process and chunk the text
# 3. Generate embeddings
# 4. Load the appropriate Gemma model based on your GPU memory
```

### Asking Questions

```python
# Simple question
answer = ask("What are the macronutrients?")
print(answer)

# With custom parameters
answer = ask(
    query="How does fiber aid digestion?",
    temperature=0.7,
    max_new_tokens=512
)

# Get answer with source context
answer, context_items = ask(
    query="What are symptoms of pellagra?",
    return_answer_only=False
)
```

### Advanced Usage

```python
# Search for relevant passages without generating an answer
scores, indices = retrieve_relevant_resources(
    query="vitamin deficiency symptoms",
    embeddings=embeddings,
    n_resources_to_return=10
)

# Print search results
print_top_results_and_scores(
    query="calcium absorption",
    embeddings=embeddings,
    n_resources_to_return=5
)
```

## System Architecture

### 1. Document Processing Pipeline
- **PDF Download**: Automatically fetches the nutrition textbook
- **Text Extraction**: Uses PyMuPDF to extract clean text
- **Sentence Segmentation**: Employs spaCy for accurate sentence boundaries
- **Chunking Strategy**: Groups sentences into 10-sentence chunks for optimal context

### 2. Embedding Generation
- **Model**: `all-mpnet-base-v2` sentence transformer
- **Batch Processing**: Efficient GPU-accelerated embedding generation
- **Storage**: Embeddings saved as CSV for persistence

### 3. Retrieval System
- **Similarity Metric**: Dot product similarity for fast retrieval
- **GPU Acceleration**: CUDA-optimized similarity calculations
- **Top-K Retrieval**: Configurable number of relevant passages

### 4. Generation Pipeline
- **Model Selection**: Automatic Gemma model selection based on GPU memory
- **Quantization**: 4-bit quantization for memory-constrained environments
- **Attention Optimization**: Flash Attention 2 support for faster inference
- **Prompt Engineering**: Structured prompts with examples for consistent output

## Configuration

### Model Selection
The system automatically selects the appropriate model based on available GPU memory:

- **< 5.1GB**: Gemma 2B with 4-bit quantization
- **5.1-8.1GB**: Gemma 2B in float16
- **8.1-19GB**: Gemma 2B in float16 or Gemma 7B in 4-bit
- **> 19GB**: Gemma 7B in float16

### Customization Options

```python
# Adjust chunking size
num_sentence_chunk_size = 10  # sentences per chunk

# Modify embedding batch size
text_chunk_embeddings = embedding_model.encode(
    text_chunks,
    batch_size=32  # adjust based on GPU memory
)

# Configure generation parameters
outputs = llm_model.generate(
    **input_ids,
    temperature=0.7,      # creativity vs consistency
    max_new_tokens=512,   # response length
    do_sample=True        # enable sampling
)
```

## Performance Optimization

### Memory Management
- Use quantization for memory-constrained environments
- Batch processing for embedding generation
- Efficient tensor operations on GPU

### Speed Optimization
- Flash Attention 2 for supported GPUs
- CUDA acceleration for similarity calculations
- Optimized chunking strategy

## Example Queries

The system excels at nutrition-related questions:

```python
queries = [
    "What are the macronutrients, and what roles do they play in the human body?",
    "How do vitamins and minerals differ in their roles and importance for health?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "How often should infants be breastfed?",
    "What role does fibre play in digestion?"
]
```

## File Structure

```
├── human-nutrition-text.pdf          # Downloaded textbook
├── text_chunks_and_embeddings_df.csv # Processed embeddings
└── nutrition_rag_system.py           # Main system code
```



