"""
BERT-based Similarity Calculator
=================================
A simple PyTorch-based information retrieval program using BERT for 
document/sentence similarity calculation.

This module provides functionality to:
- Encode text into BERT embeddings
- Calculate cosine similarity between two texts
- Find the most similar documents from a collection given a query
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class BERTSimilarityCalculator:
    """
    A class for calculating text similarity using BERT embeddings.
    
    This implementation uses DistilBERT, a smaller and faster variant of BERT,
    which is well-suited for similarity calculations while maintaining good performance.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the BERT similarity calculator.
        
        Args:
            model_name (str): Name of the pre-trained model to use.
                            Default is 'distilbert-base-uncased' for efficiency.
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into a BERT embedding vector.
        
        This method uses mean pooling over the token embeddings to create
        a fixed-size sentence representation.
        
        Args:
            text (str): The text to encode.
        
        Returns:
            np.ndarray: The embedding vector for the text.
        """
        # Tokenize the text
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to the same device as the model
        encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Use mean pooling to get sentence embedding
        # Take the mean of all token embeddings (excluding padding)
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output.last_hidden_state
        
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings and divide by the sum of mask values
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Convert to numpy and return
        return mean_pooled.cpu().numpy()
    
    def encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode multiple texts into BERT embedding vectors using batch processing.
        
        Args:
            texts (List[str]): List of texts to encode.
            batch_size (int): Number of texts to process in each batch. Default is 8.
        
        Returns:
            np.ndarray: Array of embedding vectors, shape (n_texts, embedding_dim).
        """
        embeddings = []
        
        # Process texts in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to the same device as the model
            encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
            
            # Get BERT embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Use mean pooling to get sentence embeddings
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output.last_hidden_state
            
            # Expand attention mask to match token embeddings dimensions
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings and divide by the sum of mask values
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            # Add to embeddings list
            embeddings.append(mean_pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1 (str): First text.
            text2 (str): Second text.
        
        Returns:
            float: Cosine similarity score (between -1 and 1, typically 0 to 1).
        """
        # Encode both texts
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(similarity)
    
    def find_most_similar(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Find the most similar documents to a query from a collection.
        
        Args:
            query (str): The query text.
            documents (List[str]): List of document texts to search through.
            top_k (int): Number of top similar documents to return.
        
        Returns:
            List[Tuple[int, str, float]]: List of tuples containing:
                - Document index
                - Document text
                - Similarity score
                Sorted by similarity in descending order.
        """
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Encode all documents
        print(f"Encoding {len(documents)} documents...")
        doc_embeddings = self.encode_texts(documents)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = [
            (idx, documents[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results


def main():
    """
    Example usage of the BERT Similarity Calculator.
    """
    print("=" * 80)
    print("BERT Similarity Calculator - Example Usage")
    print("=" * 80)
    print()
    
    # Initialize the calculator
    calculator = BERTSimilarityCalculator()
    print()
    
    # Example 1: Calculate similarity between two sentences
    print("-" * 80)
    print("Example 1: Calculating similarity between two sentences")
    print("-" * 80)
    
    text1 = "The cat sat on the mat"
    text2 = "A feline rested on the rug"
    text3 = "Python is a programming language"
    
    similarity_1_2 = calculator.calculate_similarity(text1, text2)
    similarity_1_3 = calculator.calculate_similarity(text1, text3)
    
    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity: {similarity_1_2:.4f}")
    print()
    print(f"Text 1: '{text1}'")
    print(f"Text 3: '{text3}'")
    print(f"Similarity: {similarity_1_3:.4f}")
    print()
    
    # Example 2: Find most similar documents from a collection
    print("-" * 80)
    print("Example 2: Finding most similar documents to a query")
    print("-" * 80)
    
    query = "machine learning and artificial intelligence"
    
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "I love cooking pasta and pizza",
        "The weather is sunny today",
        "Reinforcement learning is inspired by behavioral psychology",
        "Data science involves statistical analysis and visualization",
        "Cloud computing provides on-demand access to computing resources",
        "Blockchain technology uses distributed ledger systems"
    ]
    
    print(f"\nQuery: '{query}'")
    print(f"\nSearching through {len(documents)} documents...")
    print()
    
    results = calculator.find_most_similar(query, documents, top_k=5)
    
    print("Top 5 most similar documents:")
    print()
    for rank, (idx, doc, score) in enumerate(results, 1):
        print(f"{rank}. [Score: {score:.4f}] {doc}")
    print()
    
    # Example 3: Semantic search with different phrasings
    print("-" * 80)
    print("Example 3: Semantic search with different query phrasings")
    print("-" * 80)
    
    tech_docs = [
        "Python is a high-level programming language known for its simplicity",
        "JavaScript is widely used for web development and runs in browsers",
        "Java is an object-oriented language used for enterprise applications",
        "C++ provides low-level memory manipulation and high performance",
        "Ruby is known for its elegant syntax and the Rails framework",
        "Go is a statically typed language designed for concurrent programming",
    ]
    
    queries = [
        "simple coding language",
        "browser scripting",
        "fast system programming"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = calculator.find_most_similar(query, tech_docs, top_k=3)
        print("Top 3 matches:")
        for rank, (idx, doc, score) in enumerate(results, 1):
            print(f"  {rank}. [Score: {score:.4f}] {doc}")
    
    print()
    print("=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
