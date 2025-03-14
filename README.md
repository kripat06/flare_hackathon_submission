# Advanced RAG Chatbot

A retrieval augmented generation chatbot powered by Google Gemini, advanced vectorstore techniques, and online query capabilities.

## Overview

This application combines the power of Large Language Models (Google Gemini) with sophisticated retrieval techniques and real-time web search to create a comprehensive knowledge system. It can process various data types, analyze cryptocurrency wallets, and dynamically expand its knowledge base through web searches.

## Key Features

### Advanced RAG Implementation

- **Context Window Optimization**: Intelligently manages context windows to maximize the relevant information provided to the LLM while staying within token limits.
- **Semantic-Based Chunking**: Rather than using arbitrary chunk sizes, the system analyzes semantic breaks in content to create more meaningful document chunks.
- **Multi-Query Generation**: Automatically generates multiple subqueries from a user's question to improve retrieval accuracy and coverage.
- **Document Voting System**: Uses a consensus approach across subqueries to identify the most relevant documents.
- **Support for Multiple Data Types**: Processes text, PDFs, CSVs, and DOCX files with specialized handling for each format.

### Real-Time Knowledge Enhancement

- **Web Search Integration**: Dynamically searches the web for information not contained in the local knowledge base.
- **Cryptocurrency Analysis**: Detects wallet addresses in queries and retrieves real-time information from blockchain explorers.
- **Automatic Knowledge Base Expansion**: Adds newly discovered information to the vectorstore for future use.

### Technical Capabilities

- **Rate Limit Handling**: Implements exponential backoff and request spacing to handle API rate limits gracefully.
- **Vectorstore Management**: Automatically updates the vectorstore when new documents are added or modified.
- **Multilingual Support**: Provides responses in multiple languages based on user preference.
- **Detailed Source Attribution**: Clearly indicates the sources of information used in responses.

## How It Works

1. **Query Analysis**: When a user submits a query, the system analyzes it to determine:
   - If it contains cryptocurrency wallet addresses
   - If it requires external information not likely in the knowledge base
   - How to break it down into effective subqueries

2. **Enhanced Retrieval**:
   - Generates multiple subqueries to explore different aspects of the user's question
   - Retrieves documents for each subquery
   - Uses a voting mechanism to identify the most relevant documents
   - Splits documents based on semantic similarity rather than arbitrary character counts

3. **External Knowledge Acquisition**:
   - For cryptocurrency queries, fetches wallet information from blockchain explorers
   - For queries requiring recent or external information, performs web searches
   - Processes and extracts relevant information from search results

4. **Response Generation**:
   - Combines retrieved documents with any external information
   - Uses Google Gemini to generate a comprehensive, accurate response
   - Provides source attribution and confidence levels

5. **Knowledge Base Expansion**:
   - Automatically adds new information from web searches to the vectorstore
   - Updates the knowledge base when documents are modified

## Technical Architecture

The system consists of several key components:

- **Document Processing Pipeline**: Handles loading, chunking, and embedding of documents
- **Query Processing Module**: Generates subqueries and implements document voting
- **Web Search Enhancer**: Performs web searches and processes results
- **Cryptocurrency Analyzer**: Detects and analyzes wallet addresses
- **Vectorstore Manager**: Maintains and updates the knowledge base
- **Response Generator**: Creates final responses using the LLM

## General Architecture Diagram
<div align="center">
  <img src="https://github.com/AlaGrine/RAG_chatabot_with_Langchain/blob/main/data/docs/RAG_architecture.png" >
  <figcaption>RAG architecture with Langchain components. Credit: user AlaGrine</figcaption>
</div>

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-rag-chatbot.git
cd advanced-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GOOGLE_API_KEY="your-api-key-here"
export SERPAPI_API_KEY="your-serpapi-key-here" # Optional
```

### Running the Application

```bash
streamlit run RAG_app.py
```
