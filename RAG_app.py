####################################################################
#                         import
####################################################################

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import google_genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.schema import format_document

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Cohere
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere

# HuggingFace
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# Import streamlit
import streamlit as st

# Add to imports section
from query_processing import process_query, QueryProcessor
from web_search import enhance_with_web_search

####################################################################
#              Config: LLM services, assistant language,...
####################################################################
# Get Google API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set the GOOGLE_API_KEY environment variable")
    st.stop()

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd'hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message["english"],
        }
    ]

if "assistant_language" not in st.session_state:
    st.session_state.assistant_language = "english"

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "memory" not in st.session_state:
    st.session_state.memory = None

####################################################################
#            Create app interface with streamlit
####################################################################
st.set_page_config(page_title="Chat With Our AI Expert")

st.title("Flare X Google RAG Chatbot")

def sidebar_and_documentChooser():
    """Create the sidebar and vectorstore loader."""

    with st.sidebar:
       
        # Assistant language
        st.session_state.assistant_language = st.selectbox(
            f"Assistant language", list(dict_welcome_message.keys())
        )

    # Process documents from data directory and create/update vectorstore
    data_dir = Path(__file__).resolve().parent.joinpath("data")
    vectorstore_name = "auto_vectorstore"
    vectorstore_path = os.path.join(LOCAL_VECTOR_STORE_DIR.as_posix(), vectorstore_name)
    
    # Create directories if they don't exist
    os.makedirs(LOCAL_VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Check if we need to update the vectorstore
    should_update = False
    if not os.path.exists(vectorstore_path):
        should_update = True
    else:
        # Check if any files in data_dir are newer than the vectorstore
        vectorstore_mtime = os.path.getmtime(vectorstore_path)
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.pdf', '.txt', '.docx', '.csv')):
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) > vectorstore_mtime:
                        should_update = True
                        break
            if should_update:
                break

    if should_update:
        with st.spinner("Processing documents and updating vectorstore..."):
            try:
                # First check if there are any documents in the data directory
                data_files = []
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        if file.lower().endswith(('.pdf', '.txt', '.docx', '.csv')):
                            data_files.append(os.path.join(root, file))
                
                if not data_files:
                    st.error(f"No supported documents found in {data_dir}. Please add PDF, TXT, DOCX, or CSV files.")
                    st.stop()
                
                st.info(f"Found {len(data_files)} documents to process: {', '.join(os.path.basename(f) for f in data_files)}")

                # Copy files to temp directory
                for src_path in data_files:
                    dst_path = os.path.join(TMP_DIR, os.path.basename(src_path))
                    with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                        dst.write(src.read())

                # Load and process documents
                documents = langchain_document_loader()
                chunks = split_documents_to_chunks(documents)
                
                # Create vectorstore
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=GOOGLE_API_KEY
                )
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=vectorstore_path,
                )

                # Create retriever
                st.session_state.retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 16}
                )

                # Create memory and ConversationalRetrievalChain
                st.session_state.chain, st.session_state.memory = create_ConversationalRetrievalChain(
                    retriever=st.session_state.retriever,
                    chain_type="stuff",
                    language=st.session_state.assistant_language,
                )

                # Clear chat history
                clear_chat_history()

                st.success("Vectorstore updated successfully!")
                
            except Exception as e:
                st.error(f"Error updating vectorstore: {str(e)}")
            finally:
                # Clean up temp directory
                delte_temp_files()
    else:
        # Load existing vectorstore
        try:
            st.info("Loading existing vectorstore...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=GOOGLE_API_KEY
            )
            st.session_state.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=vectorstore_path,
            )

            # Log the number of documents in the vectorstore
            collection = st.session_state.vector_store._collection
            doc_count = collection.count()
            st.info(f"Found {doc_count} documents in the vectorstore.")

            if doc_count == 0:
                st.error("Vectorstore is empty. Please add some documents to the data directory.")
                st.stop()

            st.session_state.retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 16}
            )

            st.session_state.chain, st.session_state.memory = create_ConversationalRetrievalChain(
                retriever=st.session_state.retriever,
                chain_type="stuff",
                language=st.session_state.assistant_language,
            )

            clear_chat_history()
            st.success("Vectorstore loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")
            st.stop()


####################################################################
#        Process documents and create vectorstor (Chroma dB)
####################################################################
def delte_temp_files():
    """delete files from the './data/tmp' folder"""
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def langchain_document_loader():
    """
    Create document loaders for PDF, TXT, CSV, and DOCX files.
    https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    """
    documents = []
    loaded_files = []
    
    try:
        # Load TXT files
        txt_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
        )
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        if txt_docs:
            loaded_files.extend([doc.metadata["source"] for doc in txt_docs])
            st.info(f"Loaded {len(txt_docs)} text documents")

        # Load PDF files
        pdf_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        if pdf_docs:
            loaded_files.extend([doc.metadata["source"] for doc in pdf_docs])
            st.info(f"Loaded {len(pdf_docs)} PDF documents")

        # Load CSV files
        csv_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
            loader_kwargs={"encoding": "utf8"}
        )
        csv_docs = csv_loader.load()
        documents.extend(csv_docs)
        if csv_docs:
            loaded_files.extend([doc.metadata["source"] for doc in csv_docs])
            st.info(f"Loaded {len(csv_docs)} CSV documents")

        # Load DOCX files
        doc_loader = DirectoryLoader(
            TMP_DIR.as_posix(), glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True
        )
        docx_docs = doc_loader.load()
        documents.extend(docx_docs)
        if docx_docs:
            loaded_files.extend([doc.metadata["source"] for doc in docx_docs])
            st.info(f"Loaded {len(docx_docs)} DOCX documents")

        if not documents:
            st.error(f"No documents were successfully loaded from {TMP_DIR}")
            st.info("Files found in directory: " + ", ".join(os.listdir(TMP_DIR.as_posix())))
            raise ValueError("No documents were loaded")

        st.success(f"Successfully loaded {len(documents)} total documents")
        return documents

    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        if loaded_files:
            st.info(f"Successfully loaded files before error: {', '.join(loaded_files)}")
        raise


def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""
    try:
        st.info(f"Splitting {len(documents)} documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        st.success(f"Created {len(chunks)} chunks from the documents")
        return chunks
    except Exception as e:
        st.error(f"Error splitting documents into chunks: {str(e)}")
        raise


def create_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="semilarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-multilingual-v2.0",
    cohere_top_n=10,
):
    """
    create a retriever which can be a:
        - Vectorstore backed retriever: this is the base retriever.
        - Contextual compression retriever: We wrap the the base retriever in a ContextualCompressionRetriever.
            The compressor here is a Document Compressor Pipeline, which splits documents
            to smaller chunks, removes redundant documents, filters the top relevant documents,
            and reorder the documents so that the most relevant are at beginning / end of the list.
        - Cohere_reranker: CohereRerank endpoint is used to reorder the results based on relevance.

    Parameters:
        vector_store: Chroma vector database.
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.

        retriever_type (str): in [Vectorstore backed retriever,Contextual compression,Cohere reranker]. default = Cohere reranker

        base_retreiver_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retreiver_k: The most similar vectors are returned (default k = 16).

        compression_retriever_k: top k documents returned by the compression retriever, default = 20

        cohere_api_key: Cohere API key
        cohere_model (str): model used by Cohere, in ["rerank-multilingual-v2.0","rerank-english-v2.0"]
        cohere_top_n: top n documents returned bu Cohere, default = 10

    """

    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    elif retriever_type == "Contextual compression":
        compression_retriever = create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )
        return compression_retriever

    elif retriever_type == "Cohere reranker":
        cohere_retriever = CohereRerank_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n,
        )
        return cohere_retriever
    else:
        pass


def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    """Build a ContextualCompressionRetriever.
    We wrap the the base_retriever (a Vectorstore-backed retriever) in a ContextualCompressionRetriever.
    The compressor here is a Document Compressor Pipeline, which splits documents
    to smaller chunks, removes redundant documents, filters the top relevant documents,
    and reorder the documents so that the most relevant are at beginning / end of the list.

    Parameters:
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.
        base_retriever: a Vectorstore-backed retriever.
        chunk_size (int): Docs will be splitted into smaller chunks using a CharacterTextSplitter with a default chunk_size of 500.
        k (int): top k relevant documents to the query are filtered using the EmbeddingsFilter. default =16.
        similarity_threshold : similarity_threshold of the  EmbeddingsFilter. default =None
    """

    # 1. splitting docs into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". "
    )

    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )

    # 4. Reorder the documents

    # Less relevant document will be at the middle of the list and more relevant elements at beginning / end.
    # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
    reordering = LongContextReorder()

    # 5. create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

    return compression_retriever


def CohereRerank_retriever(
    base_retriever, cohere_api_key, cohere_model="rerank-multilingual-v2.0", top_n=10
):
    """Build a ContextualCompressionRetriever using CohereRerank endpoint to reorder the results
    based on relevance to the query.

    Parameters:
       base_retriever: a Vectorstore-backed retriever
       cohere_api_key: the Cohere API key
       cohere_model: the Cohere model, in ["rerank-multilingual-v2.0","rerank-english-v2.0"], default = "rerank-multilingual-v2.0"
       top_n: top n results returned by Cohere rerank. default = 10.
    """

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n
    )

    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return retriever_Cohere


def chain_RAG_blocks():
    """The RAG system is composed of:
    - 1. Retrieval: includes document loaders, text splitter, vectorstore and retriever.
    - 2. Memory.
    - 3. Converstaional Retreival chain.
    """
    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = []
        if not GOOGLE_API_KEY:
            error_messages.append("set the GOOGLE_API_KEY environment variable")

        if not st.session_state.uploaded_file_list:
            error_messages.append("select documents to upload")
        if st.session_state.vector_store_name == "":
            error_messages.append("provide a Vectorstore name")

        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Please "
                + ", ".join(error_messages[:-1])
                + ", and "
                + error_messages[-1]
                + "."
            )
        else:
            st.session_state.error_message = ""
            try:
                # 1. Delete old temp files
                delte_temp_files()

                # 2. Upload selected documents to temp directory
                if st.session_state.uploaded_file_list is not None:
                    for uploaded_file in st.session_state.uploaded_file_list:
                        error_message = ""
                        try:
                            temp_file_path = os.path.join(
                                TMP_DIR.as_posix(), uploaded_file.name
                            )
                            with open(temp_file_path, "wb") as temp_file:
                                temp_file.write(uploaded_file.read())
                        except Exception as e:
                            error_message += e
                    if error_message != "":
                        st.warning(f"Errors: {error_message}")

                    # 3. Load documents with Langchain loaders
                    documents = langchain_document_loader()

                    # 4. Split documents to chunks
                    chunks = split_documents_to_chunks(documents)
                    
                    # 5. Create embeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=GOOGLE_API_KEY
                    )

                    # 6. Create a vectorstore
                    persist_directory = (
                        LOCAL_VECTOR_STORE_DIR.as_posix()
                        + "/"
                        + st.session_state.vector_store_name
                    )

                    try:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                        )
                        st.info(
                            f"Vectorstore **{st.session_state.vector_store_name}** is created succussfully."
                        )

                        # 7. Create retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )

                        # 8. Create memory and ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )

                        # 9. Cclear chat_history
                        clear_chat_history()

                    except Exception as e:
                        st.error(e)

            except Exception as error:
                st.error(f"An error occurred: {error}")


####################################################################
#                       Create memory
####################################################################


def create_memory(model_name="gpt-3.5-turbo", memory_max_token=None):
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models"""
    memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
    )
    return memory


####################################################################
#          Create ConversationalRetrievalChain with memory
####################################################################


def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context
    to the `LLM` wihch will answer."""

    template = f""" You are an expert on all things cryptocurrency and blockchain.

                Generate an answer to the user query on the given context. (delimited by <context></context>). If there is no match, use your own expertise knowledge to answer the question.
                Your answer must be in the language at the end. 

                <context>
                {{chat_history}}

                {{context}} 
                </context>

                Question: {{question}}

                Language: {language}.
                """
    return template


def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="english",
):
    """Create a ConversationalRetrievalChain with Google Gemini Pro."""

    # 1. Define the standalone_question prompt.
    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""",
    )

    # 2. Define the answer_prompt
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationBufferMemory
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        input_key="question",
    )

    # 4. Instantiate LLMs with Gemini Pro
    standalone_query_generation_llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="gemini-1.5-flash",
        temperature=0.1,
        convert_system_message_to_human=True,
    )
    response_generation_llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model="gemini-1.5-flash",
        temperature=0.7,
        convert_system_message_to_human=True,
    )

    # 5. Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory


def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        # Initialize a placeholder for the spinner
        status_placeholder = st.empty()
        
        # Check if we need to enhance with web search
        status_placeholder.info("🔍 Analyzing query for external information needs...")
        vectorstore_path = os.path.join(LOCAL_VECTOR_STORE_DIR.as_posix(), "auto_vectorstore")
        
        # Detect wallet addresses first
        from web_search import WebSearchEnhancer
        enhancer = WebSearchEnhancer(GOOGLE_API_KEY)
        has_wallet, wallet_type, wallet_address = enhancer.detect_wallet_address(prompt)
        
        if has_wallet:
            status_placeholder.info(f"💰 Fetching information about {wallet_type} wallet {wallet_address}...")
            
        web_search_results = enhance_with_web_search(prompt, GOOGLE_API_KEY, vectorstore_path)
        
        # If web search was performed and documents were added, reload the vectorstore
        if web_search_results['external_search_performed']:
            if web_search_results['documents_added'] > 0:
                status_placeholder.info(f"📥 Adding {web_search_results['documents_added']} new documents from external sources...")
                # Reload vectorstore to include new documents
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=GOOGLE_API_KEY
                )
                st.session_state.vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=vectorstore_path,
                )
                st.session_state.retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 16}
                )
        
        # Process query using enhanced retrieval if retriever is available
        if st.session_state.retriever:
            try:
                # Process query using query processor
                status_placeholder.info("🧠 Generating subqueries to improve retrieval...")
                from query_processing import process_query
                query_results = process_query(prompt, st.session_state.retriever, GOOGLE_API_KEY)
            except Exception as query_error:
                print(f"Error in query processing: {str(query_error)}")
                query_results = {
                    'subqueries': [],
                    'document_votes': {},
                    'chunks': []
                }
        else:
            query_results = {
                'subqueries': [],
                'document_votes': {},
                'chunks': []
            }
        
        # Get response using original chain
        status_placeholder.info("🔎 Querying vector database for relevant documents...")
        
        # Get response using original chain
        status_placeholder.info("🤖 Generating response with Gemini...")
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        # Clear the status message
        status_placeholder.empty()
        
        # Display results
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            # If we have wallet info, display it with better formatting
            if web_search_results.get('wallet_info'):
                wallet_info = web_search_results.get('wallet_info', '')
                wallet_source = web_search_results.get('wallet_source')
                
                # Create a clean card for wallet information
                with st.container():
                    st.markdown("### 💰 Wallet Information")
                    st.markdown(wallet_info)
                    
                    if wallet_source:
                        st.markdown(f"**Source:** [{wallet_source}]({wallet_source})")
                    
                    st.markdown("---")
            
            # Display the answer
            st.markdown(answer)
            
            # Show web search results if any
            if web_search_results['search_results']:
                with st.expander("**Web Search Results**"):
                    for i, result in enumerate(web_search_results['search_results'], 1):
                        st.markdown(f"**{i}. [{result['title']}]({result['link']})**")
                        st.markdown(result['snippet'])
                        st.markdown("---")
            
            # Add subquery information in an expander if we have results
            if query_results['subqueries']:
                with st.expander("**Query Analysis**"):
                    st.write("**Generated Subqueries:**")
                    for i, subq in enumerate(query_results['subqueries'], 1):
                        st.write(f"{i}. {subq}")
                    
                    if query_results['document_votes']:
                        st.write("\n**Document Relevance Votes:**")
                        for doc_id, votes in query_results['document_votes'].items():
                            st.write(f"Document: {os.path.basename(doc_id)} - {votes} votes")
            
            # Original source documents expander
            with st.expander("**Source documents**"):
                documents_content = ""
                for document in response["source_documents"]:
                    try:
                        page = " (Page: " + str(document.metadata["page"]) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Source: "
                        + str(document.metadata["source"])
                        + page
                        + "**\n\n"
                    )
                    documents_content += document.page_content + "\n\n\n"
                st.markdown(documents_content)

    except Exception as e:
        st.warning(f"Error: {str(e)}")


####################################################################
#                         Chatbot
####################################################################
def chatbot():
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chat With Our Expert")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not GOOGLE_API_KEY:
            st.info("Please set the GOOGLE_API_KEY environment variable to continue.")
            st.stop()
        with st.spinner("Running..."):
            get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()



#Query q
#Gemini: Given query q, create K distinct subqueries
#U match each subquery to a document, and vote and see which document the queries agree om
#U use the highest voted document
#Split your text file/csv file data into a list of sentences, and use Sentence Transformers to embed each sentence
#You look at consecutive sentences and compare their embedding vectors and calculate the distance between these vectors.
#If the distance exceeds some threshold, you split the text at that point 
