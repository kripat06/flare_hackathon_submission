import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

class WebSearchEnhancer:
    def __init__(self, google_api_key: str, serpapi_key: Optional[str] = None):
        """
        Initialize the WebSearchEnhancer.
        
        Args:
            google_api_key (str): Google API key for Gemini
            serpapi_key (str, optional): SerpAPI key for web search
        """
        self.google_api_key = google_api_key
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model="gemini-1.5-pro",
            temperature=0.2
        )
        
        # Regex pattern for common cryptocurrency wallet addresses
        self.wallet_patterns = {
            'ethereum': r'0x[a-fA-F0-9]{40}',
            'bitcoin': r'(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}',
            'solana': r'[1-9A-HJ-NP-Za-km-z]{32,44}',
            'cardano': r'addr1[a-z0-9]{98}',
            'ripple': r'r[0-9a-zA-Z]{24,34}'
        }

    def detect_wallet_address(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect if text contains a cryptocurrency wallet address.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (has_wallet, wallet_type, wallet_address)
        """
        for wallet_type, pattern in self.wallet_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                return True, wallet_type, matches[0]
        return False, None, None

    def analyze_query_needs(self, query: str) -> Dict:
        """
        Analyze if the query needs external information.
        
        Args:
            query (str): The user query
            
        Returns:
            Dict: Analysis results
        """
        # First check for wallet addresses
        has_wallet, wallet_type, wallet_address = self.detect_wallet_address(query)
        
        if has_wallet:
            return {
                'needs_external_info': True,
                'reason': f'Contains {wallet_type} wallet address',
                'wallet_address': wallet_address,
                'wallet_type': wallet_type,
                'search_query': f"{wallet_type} wallet {wallet_address} transactions information"
            }
        
        # Use LLM to determine if query needs external information
        prompt = f"""Analyze the following query and determine if it likely requires recent or external information 
        that might not be in a static knowledge base.
        
        Query: {query}
        
        Return a JSON object with the following fields:
        - needs_external_info: boolean (true if external search would help)
        - reason: string (brief explanation of why)
        - search_query: string (an optimized search query to find relevant information)
        
        JSON response:"""
        
        try:
            response = self.llm.invoke(prompt)
            # Extract JSON from response
            response_text = str(response).strip()
            # Find JSON content (handle potential formatting issues)
            json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                return result
            else:
                # Fallback if JSON parsing fails
                return {
                    'needs_external_info': 'recent' in query.lower() or 'latest' in query.lower() or 'news' in query.lower(),
                    'reason': 'Fallback detection based on keywords',
                    'search_query': query
                }
        except Exception as e:
            print(f"Error analyzing query needs: {str(e)}")
            return {
                'needs_external_info': False,
                'reason': 'Error in analysis',
                'search_query': query
            }

    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        """
        Search the web using SerpAPI or a fallback method.
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            
        Returns:
            List[Dict]: Search results with title, snippet, and link
        """
        if self.serpapi_key:
            return self._search_with_serpapi(query, num_results)
        else:
            return self._search_with_fallback(query, num_results)

    def _search_with_serpapi(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search using SerpAPI."""
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "engine": "google",
            "num": num_results
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            results = []
            if "organic_results" in data:
                for result in data["organic_results"][:num_results]:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", "")
                    })
            return results
        except Exception as e:
            print(f"SerpAPI search error: {str(e)}")
            return []

    def _search_with_fallback(self, query: str, num_results: int = 3) -> List[Dict]:
        """Fallback search method using direct requests (less reliable)."""
        # This is a simplified fallback and may not work reliably due to Google's anti-scraping measures
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.select("div.g")[:num_results]:
                title_elem = result.select_one("h3")
                link_elem = result.select_one("a")
                snippet_elem = result.select_one("div.VwiC3b")
                
                if title_elem and link_elem and snippet_elem:
                    title = title_elem.get_text()
                    link = link_elem.get("href")
                    if link.startswith("/url?q="):
                        link = link.split("/url?q=")[1].split("&")[0]
                    snippet = snippet_elem.get_text()
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "link": link
                    })
            
            return results
        except Exception as e:
            print(f"Fallback search error: {str(e)}")
            return []

    def fetch_webpage_content(self, url: str) -> str:
        """
        Fetch and extract content from a webpage.
        
        Args:
            url (str): URL to fetch
            
        Returns:
            str: Extracted text content
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:10000]  # Limit to first 10000 chars to avoid very large texts
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    def get_wallet_info(self, wallet_type: str, wallet_address: str) -> str:
        """
        Get information about a cryptocurrency wallet.
        
        Args:
            wallet_type (str): Type of wallet (ethereum, bitcoin, etc.)
            wallet_address (str): The wallet address
            
        Returns:
            str: Information about the wallet
        """
        # Determine which blockchain explorer to use
        if wallet_type == 'ethereum':
            explorer_url = f"https://etherscan.io/address/{wallet_address}"
        elif wallet_type == 'bitcoin':
            explorer_url = f"https://www.blockchain.com/explorer/addresses/btc/{wallet_address}"
        elif wallet_type == 'solana':
            explorer_url = f"https://explorer.solana.com/address/{wallet_address}"
        elif wallet_type == 'cardano':
            explorer_url = f"https://cardanoscan.io/address/{wallet_address}"
        elif wallet_type == 'ripple':
            explorer_url = f"https://xrpscan.com/account/{wallet_address}"
        else:
            return f"No explorer available for {wallet_type} wallets."
        
        # Fetch content from the explorer
        content = self.fetch_webpage_content(explorer_url)
        
        # If content is too large or empty, just return the URL
        if not content or len(content) < 100:
            return f"Information about this wallet can be found at: {explorer_url}"
        
        # Use LLM to summarize the wallet information with strict formatting instructions
        prompt = f"""Summarize the following information about a {wallet_type} wallet ({wallet_address}).
        Focus on balance, transaction history, and any notable activity.
        
        IMPORTANT FORMATTING INSTRUCTIONS:
        1. Use ONLY plain text with proper spacing between words
        2. Keep all sentences under 80 characters in width
        3. Use proper line breaks between paragraphs
        4. Format lists with proper indentation and line breaks
        5. Ensure there are spaces between all words and after punctuation
        6. DO NOT use special formatting, markdown, or unusual characters
        7. Format currency values with proper spacing (e.g., "38,368.73 USD" not "38,368.73USD")
        
        Information:
        {content[:5000]}
        
        Summary:"""
        
        try:
            response = self.llm.invoke(prompt)
            # Clean up the response text
            formatted_text = str(response).strip()
            
            # Apply additional text cleanup to fix common formatting issues
            # Fix missing spaces after punctuation
            formatted_text = re.sub(r'([.,!?:;])([A-Za-z0-9])', r'\1 \2', formatted_text)
            
            # Fix missing spaces between numbers and units
            formatted_text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', formatted_text)
            
            # Fix missing spaces between words (camelCase or words stuck together)
            formatted_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', formatted_text)
            
            # Ensure proper spacing around currency symbols
            formatted_text = re.sub(r'(\d+)(\$|€|£|¥)', r'\1 \2', formatted_text)
            formatted_text = re.sub(r'(\$|€|£|¥)(\d+)', r'\1 \2', formatted_text)
            
            # Fix multiple spaces
            formatted_text = re.sub(r' {2,}', ' ', formatted_text)
            
            # Ensure proper line breaks (replace \n\n with actual line breaks)
            formatted_text = formatted_text.replace('\\n\\n', '\n\n')
            formatted_text = formatted_text.replace('\\n', '\n')
            
            # Wrap long lines to ensure they don't extend too far
            wrapped_lines = []
            for line in formatted_text.split('\n'):
                # Wrap at 80 characters
                if len(line) > 80:
                    # Use textwrap to wrap long lines
                    import textwrap
                    wrapped = textwrap.fill(line, width=80)
                    wrapped_lines.append(wrapped)
                else:
                    wrapped_lines.append(line)
            
            formatted_text = '\n'.join(wrapped_lines)
            
            return f"{formatted_text}\n\nSource: {explorer_url}"
        except Exception as e:
            print(f"Error summarizing wallet info: {str(e)}")
            return f"Information about this {wallet_type} wallet ({wallet_address}) can be found at: {explorer_url}"

    def create_documents_from_search(self, search_results: List[Dict]) -> List[Document]:
        """
        Create Document objects from search results.
        
        Args:
            search_results (List[Dict]): Search results
            
        Returns:
            List[Document]: Langchain Document objects
        """
        documents = []
        
        for i, result in enumerate(search_results):
            # Try to fetch full content
            content = self.fetch_webpage_content(result["link"])
            
            # If content fetch failed, use the snippet
            if not content:
                content = result["snippet"]
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "source": result["link"],
                    "title": result["title"],
                    "search_result": True,
                    "date_added": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            documents.append(doc)
        
        return documents

    def add_to_vectorstore(self, documents: List[Document], vectorstore_path: str) -> None:
        """
        Add documents to the vectorstore.
        
        Args:
            documents (List[Document]): Documents to add
            vectorstore_path (str): Path to the vectorstore
        """
        if not documents:
            return
        
        try:
            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            
            # Load existing vectorstore
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path
            )
            
            # Add documents
            vectorstore.add_documents(documents)
            vectorstore.persist()
            
            print(f"Added {len(documents)} documents to vectorstore at {vectorstore_path}")
        except Exception as e:
            print(f"Error adding documents to vectorstore: {str(e)}")

def format_wallet_info_for_display(wallet_info: str) -> dict:
    """
    Format wallet information for clean display in Streamlit.
    
    Args:
        wallet_info (str): Raw wallet information string
        
    Returns:
        dict: Formatted wallet information with content and source
    """
    result = {
        'content': wallet_info,
        'source': None
    }
    
    # Extract source if present
    if "Source: " in wallet_info:
        parts = wallet_info.split("Source: ")
        result['content'] = parts[0].strip()
        if len(parts) > 1:
            result['source'] = parts[1].strip()
    
    return result

def enhance_with_web_search(query: str, google_api_key: str, vectorstore_path: str) -> Dict:
    """
    Enhance a query with web search results.
    
    Args:
        query (str): User query
        google_api_key (str): Google API key
        vectorstore_path (str): Path to vectorstore
        
    Returns:
        Dict: Results including search results and any wallet information
    """
    enhancer = WebSearchEnhancer(google_api_key)
    
    # Analyze query
    analysis = enhancer.analyze_query_needs(query)
    
    results = {
        'original_query': query,
        'external_search_performed': False,
        'search_results': [],
        'wallet_info': None,
        'wallet_source': None,
        'documents_added': 0
    }
    
    # Check if query has wallet address
    has_wallet, wallet_type, wallet_address = enhancer.detect_wallet_address(query)
    
    if has_wallet:
        # Get wallet information
        st.info(f"💰 Retrieving {wallet_type} wallet information from blockchain explorer...")
        wallet_info = enhancer.get_wallet_info(wallet_type, wallet_address)
        
        # Format wallet info for display
        formatted_wallet_info = format_wallet_info_for_display(wallet_info)
        results['wallet_info'] = formatted_wallet_info['content']
        results['wallet_source'] = formatted_wallet_info['source']
        results['external_search_performed'] = True
        
        # Create document from wallet info
        wallet_doc = Document(
            page_content=wallet_info,
            metadata={
                'source': f"{wallet_type}_wallet_{wallet_address}",
                'title': f"{wallet_type.capitalize()} Wallet Information",
                'wallet_address': wallet_address,
                'date_added': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        
        # Add to vectorstore
        enhancer.add_to_vectorstore([wallet_doc], vectorstore_path)
        results['documents_added'] += 1
    
    # Perform web search if needed
    if analysis.get('needs_external_info', False):
        search_query = analysis.get('search_query', query)
        st.info(f"🌐 Searching the web for: {search_query}")
        search_results = enhancer.search_web(search_query)
        results['search_results'] = search_results
        results['external_search_performed'] = True
        
        if search_results:
            st.info(f"📄 Found {len(search_results)} relevant web pages, extracting content...")
            # Create documents from search results
            documents = enhancer.create_documents_from_search(search_results)
            
            # Add to vectorstore
            if documents:
                st.info(f"💾 Adding {len(documents)} web documents to knowledge base...")
                enhancer.add_to_vectorstore(documents, vectorstore_path)
                results['documents_added'] += len(documents)
    
    return results 