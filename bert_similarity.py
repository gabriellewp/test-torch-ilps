"""
BERT-based Document Similarity Calculator
Uses PyTorch and HuggingFace Transformers for information retrieval
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class BERTSimilarity:
    """
    A class for calculating document/sentence similarity using BERT embeddings. 
    """
    
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize the BERT model and tokenizer.
        
        Args:
            model_name (str): Name of the pretrained model to use
        """
        print(f"Loading model:  {model_name}...")
        self.tokenizer = AutoTokenizer. from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def encode(self, texts):
        """
        Encode texts into BERT embeddings using mean pooling.
        
        Args:
            texts (list or str): Text or list of texts to encode
            
        Returns:
            numpy.ndarray: Embeddings for the input texts
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize the texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v. to(self.device) for k, v in encoded_input.items()}
        
        # Get embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Mean pooling - take attention mask into account for correct averaging
        token_embeddings = model_output. last_hidden_state
        attention_mask = encoded_input['attention_mask']
        
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings and divide by mask sum
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded. sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings.cpu().numpy()
    
    def calculate_similarity(self, text1, text2):
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity score (0 to 1)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity
    
    def find_most_similar(self, query, documents, top_k=5):
        """
        Find the most similar documents to a query.
        
        Args:
            query (str): Query text
            documents (list): List of document texts
            top_k (int): Number of top results to return
            
        Returns: 
            list: List of tuples (document_index, similarity_score, document_text)
        """
        query_emb = self. encode(query)
        doc_embs = self.encode(documents)
        
        # Calculate similarities
        similarities = cosine_similarity(query_emb, doc_embs)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][: top_k]
        
        results = [
            (idx, similarities[idx], documents[idx])
            for idx in top_indices
        ]
        
        return results


def main():
    """
    Example usage of BERTSimilarity class
    """
    # Initialize the similarity calculator
    bert_sim = BERTSimilarity()
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Pairwise Similarity")
    print("="*80)
    
    text1 = "The cat sat on the mat."
    text2 = "A feline rested on the rug."
    text3 = "Python is a programming language."
    
    sim_1_2 = bert_sim.calculate_similarity(text1, text2)
    sim_1_3 = bert_sim.calculate_similarity(text1, text3)
    
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Similarity:  {sim_1_2:.4f}")
    
    print(f"\nText 1: {text1}")
    print(f"Text 3: {text3}")
    print(f"Similarity: {sim_1_3:.4f}")
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Information Retrieval")
    print("="*80)
    
    # Sample document collection
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is widely used for data science and machine learning.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "The Eiffel Tower is located in Paris, France.",
        "Cooking pasta requires boiling water and salt.",
        "Transformers revolutionized natural language processing.",
        "PyTorch and TensorFlow are popular deep learning frameworks.",
        "The Great Wall of China is a historic landmark."
    ]
    
    query = "What are popular frameworks for deep learning?"
    
    print(f"\nQuery: {query}")
    print(f"\nSearching through {len(documents)} documents...\n")
    
    results = bert_sim.find_most_similar(query, documents, top_k=5)
    
    print("Top 5 Most Similar Documents:")
    print("-" * 80)
    for rank, (idx, score, doc) in enumerate(results, 1):
        print(f"{rank}. [Score: {score:.4f}] {doc}")
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Multiple Query Comparison")
    print("="*80)
    
    queries = [
        "Tell me about neural networks",
        "What is natural language processing? ",
        "Information about landmarks"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = bert_sim. find_most_similar(query, documents, top_k=3)
        print("Top 3 matches:")
        for rank, (idx, score, doc) in enumerate(results, 1):
            print(f"  {rank}. [{score:.4f}] {doc}")


if __name__ == "__main__":
    main()