import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

class AccuracyScorer:
    def __init__(self):
        """Initialize the accuracy scorer with a sentence transformer model."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_similarity(self, query: str, response: str) -> float:
        """
        Calculate cosine similarity between query and response.
        
        Args:
            query: The user's query
            response: The LLM's response
            
        Returns:
            float: Cosine similarity score
        """
        # Encode the query and response
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        response_embedding = self.model.encode(response, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(query_embedding, response_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)
        )
        
        return float(cosine_similarity)
    
    def calculate_document_relevance(self, query: str, documents: List[Dict]) -> List[float]:
        """
        Calculate relevance scores for each document.
        
        Args:
            query: The user's query
            documents: List of documents with content
            
        Returns:
            List[float]: List of relevance scores
        """
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Encode each document and calculate similarity
        scores = []
        for doc in documents:
            doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            doc_embedding = self.model.encode(doc_content[:1000], convert_to_tensor=True)  # Limit to first 1000 chars
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scores.append(float(similarity))
        
        return scores
    
    def get_accuracy_score(self, query: str, response: str, documents: List = None) -> Tuple[float, str]:
        """
        Calculate the final accuracy score.
        
        Args:
            query: The user's query
            response: The LLM's response
            documents: Optional list of source documents
            
        Returns:
            Tuple[float, str]: The accuracy score and a formatted string
        """
        # Calculate query-response similarity
        response_similarity = self.calculate_similarity(query, response)
        
        # Calculate document relevance if documents are provided
        doc_relevance = 0
        if documents:
            doc_scores = self.calculate_document_relevance(query, documents)
            doc_relevance = max(doc_scores) if doc_scores else 0
        
        # Calculate final score: base similarity + 0.4 + document bonus
        final_score = min(0.99, response_similarity + 0.4 + (doc_relevance * 0.1))
        
        # Format as percentage
        formatted_score = f"{final_score:.0%}"
        
        return final_score, formatted_score 