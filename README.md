# RAG-powered Multi-agent Q&A Assistant

This project implements a RAG (Retrieval-Augmented Generation) based question-answering system with a basic agentic workflow that can route queries to specialized tools or retrieve information from documents.

## Features

### 1. Document-based Question Answering
- Loads and processes text documents from a `docs` directory
- Chunks documents into manageable pieces with overlap for context preservation
- Creates vector embeddings for efficient similarity search
- Retrieves relevant context based on query similarity
- Generates natural language answers based on the retrieved context

### 2. Tool-based Processing
The system includes two specialized tools:

#### Dictionary Tool
- Provides definitions for AI/ML terminology
- Supports fuzzy matching to handle spelling errors and variations
- Currently includes 15+ key AI and machine learning terms
- Triggered by "define" (e.g., "define neural network")

#### Calculator Tool
- Performs basic arithmetic operations (+, -, *, /)
- Supports special calculations like percentages, square roots, and exponents
- Handles various formats and common calculation queries
- Triggered by "calculate" (e.g., "calculate 15% of 200")

### 3. Agentic Workflow
- Analyzes user queries to determine the appropriate processing path
- Routes queries containing keywords to specialized tools
- Uses the RAG pipeline for general information retrieval
- Accepts variants and misspellings of keywords (e.g., "calcuate", "defin")

## Architecture

The system consists of several components:

1. **Document Ingestion**: Loads text files from the `docs` directory and splits them into smaller overlapping chunks.
2. **Vector Store**: Creates embeddings for text chunks and stores them in a FAISS vector index for efficient similarity search.
3. **Query Processing**: Routes queries either to tools or the RAG pipeline based on keyword detection.
4. **Answer Generation**: For RAG queries, extracts or generates answers from retrieved contexts based on question type.
5. **Tool Processing**: Handles specialized requests with dedicated functions for dictionary lookups and calculations.
6. **Command-line Interface**: Provides a simple text-based interface for interaction.

## Setup and Usage

### Prerequisites
- Python 3.7+
- Required packages:
  - faiss-cpu==1.7.4
  - numpy==1.24.3

### Installation

1. Clone the repository or download the code
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Preparing Documents

1. Create a `docs` directory in the same location as the script (if it doesn't exist already)
2. Add text files (`.txt` extension) to this directory. The system already includes:
   - `artificial_intelligence.txt` - Overview of AI concepts
   - `machine_learning.txt` - Information about machine learning
   - `neural_networks.txt` - Details about neural networks
   - `dictionary.txt` - Definitions for AI/ML terminology
   - `calculator.txt` - Examples of supported calculations

### Running the Assistant

1. Run the script:
   ```
   python rag_assistant.py
   ```
2. The system will load documents, process them, and present a prompt for questions
3. Type your question and press Enter
4. To exit, type 'exit' or 'quit'

## Example Queries

### RAG-based Queries
- "What is artificial intelligence?"
- "How do neural networks work?"
- "What are the types of machine learning?"
- "Explain deep learning"

### Dictionary Tool Queries
- "Define neural network"
- "Define reinforcement learning"
- "Define RAG"

### Calculator Tool Queries
- "Calculate 15 + 25"
- "Calculate 15% of 200"
- "Calculate sqrt(16)"
- "Calculate 2^3"

## Technical Implementation

- Uses FAISS for vector similarity search
- Employs Levenshtein distance for fuzzy matching in the dictionary tool
- Implements basic chunking with overlap to maintain context across chunks
- Uses lightweight placeholder embeddings for demonstration purposes

## Future Improvements

- Replace placeholder embedding function with a proper embedding model
- Integrate with a real LLM API like Gemini 2.0
- Add more specialized tools and expand existing tool functionality
- Improve the answer extraction logic for complex queries
- Develop a web interface for easier interaction
- Add document metadata extraction for better context retrieval
- Implement streaming responses for large text generations 