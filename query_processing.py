from typing import List, Dict
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st

class QueryProcessor:
    def __init__(self, google_api_key: str, k_subqueries: int = 3, similarity_threshold: float = 0.7):
        """
        Initialize the QueryProcessor.
        
        Args:
            google_api_key (str): Google API key for Gemini
            k_subqueries (int): Number of subqueries to generate
            similarity_threshold (float): Threshold for sentence splitting
        """
        self.k_subqueries = k_subqueries
        self.similarity_threshold = similarity_threshold
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model="gemini-1.5-pro",
            temperature=0.3
        )

    def generate_subqueries(self, main_query: str) -> List[str]:
        """Generate K distinct subqueries from the main query using Gemini."""
        prompt = f"""Given the following query, create {self.k_subqueries} distinct but related subqueries 
        that could help gather comprehensive information. Return only the numbered subqueries, one per line.
        
        Query: {main_query}
        """
        
        response = self.llm.invoke(prompt)
        # Split response into lines and clean up
        subqueries = [line.strip() for line in str(response).split('\n') 
                     if line.strip() and not line.strip().isdigit()]
        return subqueries[:self.k_subqueries]  # Ensure we only return k subqueries

    def vote_on_documents(self, subqueries: List[str], documents: List[dict]) -> dict:
        """Match subqueries to documents and vote on the most relevant ones."""
        document_votes = Counter()
        
        # Get embeddings for subqueries and documents
        subquery_embeddings = self.sentence_transformer.encode(subqueries)
        doc_embeddings = self.sentence_transformer.encode([doc['content'] for doc in documents])
        
        # For each subquery, find the most similar document
        for subquery_embedding in subquery_embeddings:
            similarities = np.dot(doc_embeddings, subquery_embedding)
            most_similar_idx = np.argmax(similarities)
            document_votes[documents[most_similar_idx]['id']] += 1
        
        # Return documents sorted by votes
        return dict(document_votes.most_common())

    def split_text_by_similarity(self, text: str) -> List[str]:
        """Split text into chunks based on semantic similarity between consecutive sentences."""
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return sentences
        
        # Get embeddings for all sentences
        embeddings = self.sentence_transformer.encode(sentences)
        
        # Initialize chunks
        chunks = []
        current_chunk = [sentences[0]]
        
        # Compare consecutive sentences
        for i in range(1, len(sentences)):
            similarity = np.dot(embeddings[i], embeddings[i-1])
            
            if similarity < self.similarity_threshold:
                # Start new chunk if similarity is below threshold
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            
            current_chunk.append(sentences[i])
        
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

def process_query(query: str, retriever, google_api_key: str) -> Dict:
    """
    Main function to process a query using the enhanced retrieval method.
    
    Args:
        query (str): The main query
        retriever: The retriever object from langchain
        google_api_key (str): Google API key for Gemini
    
    Returns:
        Dict: Results including subqueries and voted documents
    """
    processor = QueryProcessor(google_api_key)
    
    # Generate subqueries
    st.info("🧩 Generating diverse subqueries to improve retrieval...")
    subqueries = processor.generate_subqueries(query)
    
    # Retrieve documents for each subquery
    all_docs = []
    for i, subquery in enumerate(subqueries, 1):
        st.info(f"🔍 Retrieving documents for subquery {i}/{len(subqueries)}: '{subquery}'")
        docs = retriever.get_relevant_documents(subquery)
        all_docs.extend(docs)
    
    # Vote on documents
    st.info("🗳️ Ranking documents by relevance across all subqueries...")
    voted_documents = processor.vote_on_documents(subqueries, all_docs)
    
    # Process the top voted document if available
    chunks = []
    if voted_documents and all_docs:
        top_doc_id = list(voted_documents.keys())[0]
        # Find the document with matching source
        top_docs = [doc for doc in all_docs if doc.metadata.get('source') == top_doc_id]
        if top_docs:
            st.info("✂️ Analyzing semantic structure of top document...")
            chunks = processor.split_text_by_similarity(top_docs[0].page_content)
    
    return {
        'original_query': query,
        'subqueries': subqueries,
        'document_votes': voted_documents,
        'chunks': chunks
    } 