import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from langchain.schema import Document
import re

class AccuracyEvaluator:
    """
    Evaluates the accuracy of LLM responses compared to source documents.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the AccuracyEvaluator.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing extra whitespace, citations, etc.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Remove citations like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for more granular comparison.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting - could be improved with nltk
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if len(s) > 10]  # Filter out very short sentences
    
    def calculate_similarity(self, response: str, documents: List[Document]) -> Dict:
        """
        Calculate similarity between response and source documents.
        
        Args:
            response (str): The LLM's response
            documents (List[Document]): Source documents
            
        Returns:
            Dict: Similarity scores and analysis
        """
        # Preprocess response
        clean_response = self.preprocess_text(response)
        response_sentences = self.split_into_sentences(clean_response)
        
        if not response_sentences:
            return {
                "accuracy_score": 0.0,
                "confidence": "Low",
                "analysis": "Could not analyze response (no valid sentences found)."
            }
        
        # Preprocess documents
        doc_texts = []
        for doc in documents:
            clean_doc = self.preprocess_text(doc.page_content)
            doc_texts.append(clean_doc)
        
        combined_doc_text = " ".join(doc_texts)
        doc_sentences = self.split_into_sentences(combined_doc_text)
        
        if not doc_sentences:
            return {
                "accuracy_score": 0.0,
                "confidence": "Low",
                "analysis": "Could not analyze documents (no valid sentences found)."
            }
        
        # Get embeddings
        response_embeddings = self.model.encode(response_sentences)
        doc_embeddings = self.model.encode(doc_sentences)
        
        # Calculate sentence-level similarities
        sentence_scores = []
        for resp_emb in response_embeddings:
            # Find max similarity with any document sentence
            similarities = np.dot(doc_embeddings, resp_emb)
            max_sim = np.max(similarities)
            sentence_scores.append(max_sim)
        
        # Calculate overall accuracy score
        accuracy_score = np.mean(sentence_scores)
        
        # Determine confidence level
        if accuracy_score >= 0.85:
            confidence = "Very High"
        elif accuracy_score >= 0.75:
            confidence = "High"
        elif accuracy_score >= 0.65:
            confidence = "Moderate"
        elif accuracy_score >= 0.5:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        # Calculate percentage of sentences with good support
        well_supported = sum(1 for score in sentence_scores if score >= 0.7)
        support_percentage = (well_supported / len(sentence_scores)) * 100 if sentence_scores else 0
        
        # Prepare analysis
        analysis = f"{support_percentage:.1f}% of response statements are well-supported by the documents."
        
        return {
            "accuracy_score": float(accuracy_score),
            "confidence": confidence,
            "sentence_scores": sentence_scores,
            "analysis": analysis
        }
    
    def get_accuracy_badge(self, score: float) -> str:
        """
        Get an HTML/Markdown badge for the accuracy score.
        
        Args:
            score (float): Accuracy score
            
        Returns:
            str: Markdown for the badge
        """
        if score >= 0.85:
            color = "success"
        elif score >= 0.7:
            color = "primary"
        elif score >= 0.5:
            color = "warning"
        else:
            color = "danger"
            
        return f"<span style='display:inline-block; padding:3px 6px; border-radius:3px; background-color:{self._get_color(color)}; color:white; font-size:0.8em;'>Accuracy: {score:.2f}</span>"
    
    def _get_color(self, color_name: str) -> str:
        """Get the hex color for a named color."""
        colors = {
            "success": "#28a745",
            "primary": "#007bff",
            "warning": "#ffc107",
            "danger": "#dc3545"
        }
        return colors.get(color_name, "#6c757d")  # Default to secondary color 