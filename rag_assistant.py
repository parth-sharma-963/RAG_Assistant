import os
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


# Data Ingestion
def load_documents(docs_dir: str = 'docs') -> Dict[str, str]:
    """
    Load documents from the specified directory.
    
    Args:
        docs_dir: Path to the directory containing documents
        
    Returns:
        Dictionary mapping document names to their content
    """
    documents = {}
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Created '{docs_dir}' directory. Please add text files before running again.")
        return documents
        
    for filename in os.listdir(docs_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    
    return documents


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 20) -> List[str]:
    """
    Split text into chunks of specified size with overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Approximate number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    i = 0
    while i < len(words):
        chunk_end = min(i + chunk_size, len(words))
        chunk = ' '.join(words[i:chunk_end])
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks


def process_documents(documents: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Process documents and create text chunks with source information.
    
    Args:
        documents: Dictionary mapping document names to their content
        
    Returns:
        List of (chunk, source) tuples
    """
    all_chunks = []
    
    for doc_name, content in documents.items():
        chunks = chunk_text(content)
        for chunk in chunks:
            all_chunks.append((chunk, doc_name))
    
    return all_chunks


# Vector Store & Retrieval
def create_placeholder_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Create a placeholder embedding for a text.
    
    Args:
        text: Text to create embedding for
        dim: Dimensionality of the embedding
        
    Returns:
        Numpy array representing the embedding
    """
    # Use hash of the text to create a deterministic but pseudo-random embedding
    np.random.seed(hash(text) % 2**32)
    embedding = np.random.rand(dim).astype(np.float32)
    # Normalize to unit length
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


class VectorStore:
    def __init__(self, dim: int = 128):
        """Initialize the vector store with FAISS index.
        
        Args:
            dim: Dimensionality of the embeddings
        """
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
        self.sources = []
    
    def add_documents(self, processed_chunks: List[Tuple[str, str]]):
        """
        Add documents to the vector store.
        
        Args:
            processed_chunks: List of (chunk, source) tuples
        """
        if not processed_chunks:
            return
            
        embeddings = []
        
        for chunk, source in processed_chunks:
            embedding = create_placeholder_embedding(chunk)
            embeddings.append(embedding)
            self.chunks.append(chunk)
            self.sources.append(source)
        
        embeddings_array = np.array(embeddings).astype(np.float32)
        self.index.add(embeddings_array)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most similar chunks to the query.
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of dictionaries containing retrieved chunks and their metadata
        """
        if self.index.ntotal == 0:
            return []
            
        query_embedding = create_placeholder_embedding(query)
        query_embedding = np.array([query_embedding]).astype(np.float32)
        
        # Adjust top_k to the number of available chunks
        actual_top_k = min(top_k, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, actual_top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS returns -1 if fewer than top_k results available
                results.append({
                    'chunk': self.chunks[idx],
                    'source': self.sources[idx],
                    'distance': float(distances[0][i])
                })
        
        return results


# LLM Integration
def generate_answer(query: str, context: List[Dict[str, Any]]) -> str:
    """
    Generate an answer based on the query and retrieved context.
    
    Args:
        query: User's query
        context: Retrieved context chunks
        
    Returns:
        Generated answer
    """
    if not context:
        return "I don't have enough information to answer that question."
    
    # Process the query to identify question type
    query_lower = query.lower().strip()
    
    # For "what is" questions - try to find a definition
    if query_lower.startswith("what is") or query_lower.startswith("what are") or "whatis" in query_lower:
        topic = query_lower.replace("what is", "").replace("what are", "").replace("whatis", "").strip()
        
        # First, search for exact topic match in context
        for ctx in context:
            if ctx['source'] == 'artificial_intelligence.txt' and 'artificial intelligence' in topic:
                sentences = ctx['chunk'].split('.')
                for sentence in sentences:
                    if 'artificial intelligence' in sentence.lower() and any(word in sentence.lower() for word in ["is", "are", "refers"]):
                        return sentence.strip() + "."
            
            if ctx['source'] == 'neural_networks.txt' and 'neural network' in topic:
                sentences = ctx['chunk'].split('.')
                for sentence in sentences:
                    if 'neural network' in sentence.lower() and any(word in sentence.lower() for word in ["is", "are", "refers"]):
                        return sentence.strip() + "."
                        
            if ctx['source'] == 'machine_learning.txt' and 'machine learning' in topic:
                sentences = ctx['chunk'].split('.')
                for sentence in sentences:
                    if 'machine learning' in sentence.lower() and any(word in sentence.lower() for word in ["is", "are", "refers"]):
                        return sentence.strip() + "."
        
        # Check all contexts for the topic
        for ctx in context:
            text = ctx['chunk'].lower()
            if topic in text:
                sentences = ctx['chunk'].split('.')
                for sentence in sentences:
                    if topic in sentence.lower() and any(word in sentence.lower() for word in ["is", "are", "refers"]):
                        return sentence.strip() + "."
    
    # For "how" questions about processes
    if query_lower.startswith("how") and any(word in query_lower for word in ["work", "function", "do", "does", "process"]):
        # Special handling for neural network questions
        if 'neural network' in query_lower:
            for ctx in context:
                if ctx['source'] == 'neural_networks.txt' and ('structure' in ctx['chunk'].lower() or 'function' in ctx['chunk'].lower()):
                    return ctx['chunk']
                
        # Check for process descriptions in context
        for ctx in context:
            if any(term in ctx['chunk'].lower() for term in ["process", "works", "function", "steps"]):
                return ctx['chunk']
    
    # For questions about types or categories
    if any(word in query_lower for word in ["types", "kinds", "categories"]):
        for ctx in context:
            if any(term in ctx['chunk'].lower() for term in ["types", "kinds", "categories"]) or \
               ":" in ctx['chunk'] or any(f"{i}." in ctx['chunk'] for i in range(1, 10)):
                return ctx['chunk']
    
    # Default to the most relevant context - if it's very short, combine the top two
    if len(context) > 1 and len(context[0]['chunk']) < 200:
        return context[0]['chunk'] + "\n\n" + context[1]['chunk']
    
    return context[0]['chunk']


# Tool-based Processing
def use_tool(query: str) -> str:
    """
    Function for tool-based processing.
    
    Args:
        query: User's query
        
    Returns:
        Result of the tool processing
    """
    query_lower = query.lower().strip()
    
    # Handle define queries and "what is" questions for the dictionary tool
    if "define" in query_lower or query_lower.startswith("what is") or query_lower.startswith("what are"):
        # Extract the term to define
        if "define" in query_lower:
            term = query_lower.replace("define", "").strip()
        elif query_lower.startswith("what is"):
            term = query_lower.replace("what is", "").strip()
        elif query_lower.startswith("what are"):
            term = query_lower.replace("what are", "").strip()
        else:
            term = ""
        
        # Dictionary of predefined terms
        definitions = {
            "artificial intelligence": "Artificial Intelligence (AI) is a field of computer science focused on creating systems capable of performing tasks that typically require human intelligence.",
            "ai": "Artificial Intelligence (AI) is a field of computer science focused on creating systems capable of performing tasks that typically require human intelligence.",
            "machine learning": "Machine Learning is a subset of artificial intelligence that enables computers to learn from data and improve from experience without being explicitly programmed.",
            "ml": "Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn from data and improve from experience without being explicitly programmed.",
            "neural network": "Neural Networks are computational models inspired by the human brain's structure and function that form the foundation of deep learning.",
            "neural networks": "Neural Networks are computational models inspired by the human brain's structure and function that form the foundation of deep learning.",
            "nn": "Neural Networks (NN) are computational models inspired by the human brain's structure and function that form the foundation of deep learning.",
            "deep learning": "Deep Learning is a subfield of machine learning that uses neural networks with many layers to analyze various factors of data.",
            "dl": "Deep Learning (DL) is a subfield of machine learning that uses neural networks with many layers to analyze various factors of data.",
            "supervised learning": "Supervised Learning is a machine learning approach where the algorithm is trained on a labeled dataset to predict outputs based on input features.",
            "unsupervised learning": "Unsupervised Learning is a machine learning approach where the algorithm is given unlabeled data and must find patterns and relationships within it.",
            "natural language processing": "Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language.",
            "nlp": "Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language.",
            "computer vision": "Computer Vision is a field of AI that enables computers to derive meaningful information from digital images and videos.",
            "cv": "Computer Vision (CV) is a field of AI that enables computers to derive meaningful information from digital images and videos.",
            "reinforcement learning": "Reinforcement Learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward.",
            "rl": "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward.",
            "generative ai": "Generative AI refers to AI systems that can generate new content, such as text, images, audio, or video, that wasn't explicitly programmed.",
            "large language model": "A Large Language Model (LLM) is a type of deep learning model trained on vast amounts of text data, capable of understanding and generating human language.",
            "llm": "A Large Language Model (LLM) is a type of deep learning model trained on vast amounts of text data, capable of understanding and generating human language.",
            "transfer learning": "Transfer Learning is a machine learning method where a model developed for one task is reused as the starting point for a model on a second task.",
            "fine-tuning": "Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task or domain.",
            "semi-supervised learning": "Semi-supervised Learning is a machine learning approach that combines a small amount of labeled data with a large amount of unlabeled data during training.",
            "rag": "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of relevant information from a knowledge base with generation of text using a language model."
        }
        
        # Function to compute Levenshtein edit distance between two strings
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
                
            if len(s2) == 0:
                return len(s1)
                
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
                
            return previous_row[-1]
        
        # Convert Levenshtein distance to a similarity score
        def string_similarity(s1, s2):
            # Convert to lowercase and remove leading/trailing spaces
            s1, s2 = s1.lower().strip(), s2.lower().strip()
            
            # If exact match, return perfect score
            if s1 == s2:
                return 1.0
                
            # Calculate Levenshtein distance
            distance = levenshtein_distance(s1, s2)
            max_len = max(len(s1), len(s2))
            
            if max_len == 0:
                return 0
                
            # Convert distance to similarity score (1 - normalized_distance)
            return 1 - (distance / max_len)
        
        # Try to find exact match first
        for key, value in definitions.items():
            if term == key:
                return value
        
        # If no exact match, look for partial matches
        for key, value in definitions.items():
            if term in key or key in term:
                return value
        
        # If still no match, look for closest match using Levenshtein distance
        best_match = None
        best_score = 0.7  # Threshold for minimum similarity
        
        for key in definitions.keys():
            score = string_similarity(term, key)
            if score > best_score:
                best_score = score
                best_match = key
        
        if best_match:
            return f"I found a close match: '{best_match}'\n\n{definitions[best_match]}"
        
        return f"I don't have a definition for '{term}' in my knowledge base."
    
    # Handle calculate queries - calculator tool
    elif "calculate" in query_lower:
        # Extract the expression to calculate
        expression = query_lower.replace("calculate", "").strip()
        
        # Handle simple arithmetic operations
        try:
            # Handle percentage calculations
            if "%" in expression and "of" in expression:
                parts = expression.split("of")
                if len(parts) == 2:
                    percent_part = parts[0].strip().replace("%", "")
                    value_part = parts[1].strip()
                    
                    percent = float(percent_part) / 100
                    value = float(value_part)
                    result = percent * value
                    return f"{percent_part}% of {value_part} = {result}"
                
            # Handle square root
            elif "sqrt" in expression:
                import re
                match = re.search(r"sqrt\s*\(\s*(\d+\.?\d*)\s*\)", expression)
                if match:
                    num = float(match.group(1))
                    import math
                    result = math.sqrt(num)
                    return f"Square root of {num} = {result}"
            
            # Handle power operations with ^
            elif "^" in expression:
                base, exponent = expression.split("^")
                base = float(base.strip())
                exponent = float(exponent.strip())
                result = base ** exponent
                return f"{base} raised to the power of {exponent} = {result}"
            
            # Evaluate simple arithmetic expressions
            else:
                # Replace multiply symbol * with Python's multiplication operator *
                expression = expression.replace("×", "*").replace("x", "*")
                # Use Python's eval() to calculate the result
                result = eval(expression)
                return f"{expression} = {result}"
        
        except Exception as e:
            return f"I couldn't calculate the expression: {expression}. Please check the format and try again."
    
    return f"Processed by tool: {query}"


# Agent Workflow
def process_query(query: str, vector_store: VectorStore) -> Dict[str, Any]:
    """
    Process a user query using either a tool or the RAG pipeline.
    
    Args:
        query: User's query
        vector_store: Vector store for retrieval
        
    Returns:
        Dictionary containing processing results
    """
    # Check if query should be routed to a tool
    query_lower = query.lower().strip()
    
    # Tool keywords with common spelling variations
    tool_mappings = {
        "calculate": ["calculate", "compute", "calcuate", "calculat", "calc"],
        "define": ["define", "definition", "meaning", "defin", "what is", "what are"]
    }
    
    # Check for tool keywords with spelling tolerance
    for tool_type, variations in tool_mappings.items():
        for variation in variations:
            if variation in query_lower:
                print(f"Routing to {tool_type} tool")
                tool_response = use_tool(query)
                return {
                    "agent_path": "tool",
                    "context": None,
                    "answer": tool_response
                }
    
    # If not, use the RAG pipeline
    print("Routing to RAG pipeline")
    context = vector_store.retrieve(query)
    answer = generate_answer(query, context)
    
    return {
        "agent_path": "rag",
        "context": context,
        "answer": answer
    }


# CLI Interface
def run_cli():
    """Run the command-line interface for the Q&A assistant."""
    print("Loading documents...")
    documents = load_documents()
    
    if not documents:
        print("No documents found in the 'docs' directory. Please add some .txt files and run again.")
        return
        
    print(f"Loaded {len(documents)} documents.")
    
    print("Processing documents...")
    processed_chunks = process_documents(documents)
    
    print("Building vector store...")
    vector_store = VectorStore()
    vector_store.add_documents(processed_chunks)
    
    print(f"Added {len(processed_chunks)} text chunks to the vector store.")
    print("\n--- RAG-powered Q&A Assistant ---")
    print("Type 'exit' to quit the program.")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        result = process_query(query, vector_store)
        
        print(f"\nAgent path: {result['agent_path']}")
        
        if result['context']:
            print("\nRetrieved context:")
            for i, ctx in enumerate(result['context'], 1):
                print(f"{i}. From {ctx['source']}: {ctx['chunk'][:100]}...")
        
        print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    run_cli() 