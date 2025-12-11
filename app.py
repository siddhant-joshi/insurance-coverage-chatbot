"""
RAG-based Insurance Policy Chatbot
Answers questions about UnitedHealthcare commercial medical policies.
"""

import os
import shutil
import requests
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

DATA_DIR = Path("data")
VECTOR_STORE_PATH = DATA_DIR / "vector_store"

SYSTEM_PROMPT = """You are an expert insurance policy analyst. Answer the user's question based ONLY on the provided context from UnitedHealthcare policy documents.

Context:
{context}

Instructions:
- Provide accurate, detailed answers based solely on the policy documents.
- If specific coverage criteria, exclusions, or requirements are mentioned, include them.
- If the information is not in the context, clearly state "I don't have enough information in the available policy documents to answer this question."
- Always be clear about which policy document the information comes from when possible.

Question: {question}

Answer:"""


def downloadPolicies(urls: List[str], dataDir: Path) -> List[Path]:
    """
    Download PDF policy documents from the provided URLs.
    
    Args:
        urls: List of URLs pointing to PDF documents.
        dataDir: Directory to save downloaded PDFs.
    
    Returns:
        List of paths to downloaded PDF files.
    """
    dataDir.mkdir(parents=True, exist_ok=True)
    downloadedFiles = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    for url in urls:
        fileName = url.split("/")[-1]
        filePath = dataDir / fileName
        
        if filePath.exists() and filePath.stat().st_size > 0:
            downloadedFiles.append(filePath)
            continue
        
        maxRetries = 3
        for attempt in range(maxRetries):
            try:
                response = requests.get(
                    url, 
                    headers=headers, 
                    timeout=60,
                    stream=True
                )
                response.raise_for_status()
                
                with open(filePath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloadedFiles.append(filePath)
                break
                
            except requests.RequestException:
                if attempt < maxRetries - 1:
                    continue
    
    return downloadedFiles


def loadAndProcessDocuments(pdfPaths: List[Path]) -> List:
    """
    Load PDF documents and split them into chunks.
    
    Args:
        pdfPaths: List of paths to PDF files.
    
    Returns:
        List of document chunks.
    """
    allDocuments = []
    
    for pdfPath in pdfPaths:
        try:
            loader = PyPDFLoader(str(pdfPath))
            documents = loader.load()
            
            for doc in documents:
                doc.metadata["source"] = pdfPath.name
            
            allDocuments.extend(documents)
            
        except Exception as e:
            st.error(f"Error loading {pdfPath.name}: {str(e)}")
    
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = textSplitter.split_documents(allDocuments)
    
    return chunks


@st.cache_resource(show_spinner=False)
def createVectorStore(_chunks: List) -> FAISS:
    """
    Create FAISS vector store from document chunks.
    Uses HuggingFace embeddings (all-MiniLM-L6-v2).
    
    Args:
        _chunks: List of document chunks.
    
    Returns:
        FAISS vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    vectorStore = FAISS.from_documents(_chunks, embeddings)
    
    return vectorStore


@st.cache_resource(show_spinner=False)
def initializeEmbeddings():
    """
    Initialize and cache HuggingFace embeddings model.
    
    Returns:
        HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def loadOrCreateVectorStore(chunks: List, embeddings) -> FAISS:
    """
    Load existing vector store or create a new one.
    
    Args:
        chunks: List of document chunks.
        embeddings: Embeddings model instance.
    
    Returns:
        FAISS vector store instance.
    """
    if VECTOR_STORE_PATH.exists():
        try:
            vectorStore = FAISS.load_local(
                str(VECTOR_STORE_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorStore
        except Exception:
            pass
    
    vectorStore = FAISS.from_documents(chunks, embeddings)
    
    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    vectorStore.save_local(str(VECTOR_STORE_PATH))
    
    return vectorStore


def formatDocs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def createRagChain(vectorStore: FAISS):
    """
    Create the RAG chain with Gemini LLM using LCEL.
    
    Args:
        vectorStore: FAISS vector store for retrieval.
    
    Returns:
        Tuple of (retriever, chain, llm).
    """
    apiKey = os.getenv("GOOGLE_API_KEY")
    
    if not apiKey:
        st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
        st.stop()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=apiKey,
        temperature=0.3
    )
    
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert insurance policy analyst. Answer the user's question based ONLY on the provided context from UnitedHealthcare policy documents.

Context:
{context}

Instructions:
- Provide accurate, detailed answers based solely on the policy documents.
- If specific coverage criteria, exclusions, or requirements are mentioned, include them.
- If the information is not in the context, clearly state "I don't have enough information in the available policy documents to answer this question."
- Always be clear about which policy document the information comes from when possible."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    chain = (
        {
            "context": lambda x: formatDocs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return retriever, chain


def getExistingPdfs() -> List[Path]:
    """Get list of existing PDFs in data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return list(DATA_DIR.glob("*.pdf"))


def processUploadedFiles(uploadedFiles) -> List[Path]:
    """
    Save uploaded PDF files to data directory.
    
    Args:
        uploadedFiles: List of uploaded file objects from Streamlit.
    
    Returns:
        List of paths to saved PDF files.
    """
    savedPaths = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for uploadedFile in uploadedFiles:
        filePath = DATA_DIR / uploadedFile.name
        with open(filePath, "wb") as f:
            f.write(uploadedFile.getbuffer())
        savedPaths.append(filePath)
    
    return savedPaths


def initializePipelineFromPdfs(pdfPaths: List[Path]):
    """
    Initialize the RAG pipeline from PDF paths.
    
    Args:
        pdfPaths: List of paths to PDF files.
    
    Returns:
        Tuple of (vector_store, embeddings, chunk_count).
    """
    with st.spinner("Processing documents..."):
        chunks = loadAndProcessDocuments(pdfPaths)
    
    if not chunks:
        st.error("No content could be extracted from the documents.")
        return None, None, 0
    
    with st.spinner("Creating vector embeddings (this may take a moment)..."):
        embeddings = initializeEmbeddings()
        vectorStore = loadOrCreateVectorStore(chunks, embeddings)
    
    return vectorStore, embeddings, len(chunks)


def renderChatMessage(role: str, content: str, sources: List = None, isError: bool = False):
    """
    Render a chat message with optional source citations.
    
    Args:
        role: Message role ('user' or 'assistant').
        content: Message content.
        sources: Optional list of source documents.
        isError: Whether this message is an error message.
    """
    avatar = "👤" if role == "user" else "📋"
    with st.chat_message(role, avatar=avatar):
        if isError:
            st.markdown(f'<p style="color: #ff4b4b; font-weight: 500;">{content}</p>', unsafe_allow_html=True)
        else:
            st.markdown(content)
        
        if sources and role == "assistant":
            uniqueSources = list(set([doc.metadata.get("source", "Unknown") for doc in sources]))
            if uniqueSources:
                with st.expander("View Sources"):
                    for source in uniqueSources:
                        st.caption(f"• {source}")


def main():
    """Main application entry point."""
    
    st.set_page_config(
        page_title="Insurance Policy Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        section[data-testid="stSidebar"] {
            font-size: 0.85rem;
            width: 35rem !important;
            min-width: 35rem !important;
            max-width: 35rem !important;
            transform: none !important;
            transition: none !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 35rem !important;
            min-width: 35rem !important;
            max-width: 35rem !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"],
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 35rem !important;
            min-width: 35rem !important;
            max-width: 35rem !important;
            transform: translateX(0) !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] > div,
        section[data-testid="stSidebar"][aria-expanded="false"] > div {
            width: 35rem !important;
            min-width: 35rem !important;
            max-width: 35rem !important;
        }
        button[kind="header"] {
            display: none !important;
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.2rem;
        }
        [data-testid="stSidebar"] h3 {
            font-size: 1rem;
        }
        [data-testid="stSidebar"] .stMarkdown {
            font-size: 0.85rem;
        }
        [data-testid="stSidebar"] .stButton button {
            font-size: 0.85rem;
            padding: 0.35rem 0.75rem;
        }
        [data-testid="stSidebar"] .stTextInput input {
            font-size: 0.85rem;
        }
        [data-testid="stSidebar"] .stCaption {
            font-size: 0.75rem;
        }
        .fixed-header {
            position: fixed;
            top: 3.5rem;
            left: 35rem !important;
            right: 0;
            background-color: #0E1117;
            z-index: 999;
            padding: 1rem 1rem 0.5rem 1rem;
            border-bottom: 1px solid #262730;
        }
        .fixed-header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            margin: 0;
            padding: 0;
            color: #FAFAFA;
        }
        .fixed-header p {
            font-size: 0.875rem;
            color: #A3A8B4;
            margin: 0.25rem 0 0 0;
            padding: 0;
        }
        .main-content {
            padding-top: 9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="fixed-header" id="main-header">
            <h1>Insurance Policy Assistant</h1>
            <p>Ask Questions About UnitedHealthcare Commercial Medical Policies</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Document Management")
        
        with st.expander("Upload PDFs", expanded=True):
            st.markdown("""
            **Steps to add policy documents:**
            
            1. Visit the [UHC Policy Portal](https://www.uhcprovider.com/en/policies-protocols/commercial-policies/commercial-medical-drug-policies.html) in Safari
            2. Accept Terms & Conditions when prompted
            3. Download the policy PDFs you need (Cmd+Click to select multiple)
            4. Upload them using the file selector below
            """)
            
            if "uploader_key" not in st.session_state:
                st.session_state.uploader_key = 0
            
            uploadedFiles = st.file_uploader(
                "Select PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"pdf_uploader_{st.session_state.uploader_key}"
            )
            
            if uploadedFiles:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Process Uploaded PDFs", use_container_width=True):
                        processUploadedFiles(uploadedFiles)
                        st.session_state.pop("vector_store", None)
                        st.session_state.pop("rag_chain", None)
                        st.session_state.pop("retriever", None)
                        if VECTOR_STORE_PATH.exists():
                            shutil.rmtree(VECTOR_STORE_PATH)
                        st.session_state.uploader_key += 1
                        st.rerun()
        
        existingPdfs = getExistingPdfs()
        if existingPdfs:
            st.markdown(f"### Loaded Documents ({len(existingPdfs)})")
            with st.container(height=200):
                for pdf in existingPdfs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(pdf.name)
                    with col2:
                        if st.button("Delete", key=f"del_{pdf.name}", help=f"Delete {pdf.name}"):
                            pdf.unlink()
                            st.session_state.pop("vector_store", None)
                            st.session_state.pop("rag_chain", None)
                            st.session_state.pop("retriever", None)
                            if VECTOR_STORE_PATH.exists():
                                shutil.rmtree(VECTOR_STORE_PATH)
                            st.rerun()
        else:
            st.info("No documents loaded yet")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.pop("rag_chain", None)
                st.session_state.pop("retriever", None)
                st.rerun()
        
        with col2:
            if st.button("Rebuild Index", use_container_width=True):
                st.session_state.pop("vector_store", None)
                st.session_state.pop("rag_chain", None)
                st.session_state.pop("retriever", None)
                if VECTOR_STORE_PATH.exists():
                    shutil.rmtree(VECTOR_STORE_PATH)
                st.rerun()
    
    existingPdfs = getExistingPdfs()
    
    if not existingPdfs:
        st.info("Upload policy PDFs using the sidebar to get started")
        st.stop()
    
    if "vector_store" not in st.session_state:
        vectorStore, embeddings, chunkCount = initializePipelineFromPdfs(existingPdfs)
        if vectorStore is None:
            st.stop()
        st.session_state.vector_store = vectorStore
        st.session_state.chunk_count = chunkCount
    
    # st.success(f"Loaded {len(existingPdfs)} policy document(s)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "rag_chain" not in st.session_state:
        retriever, chain = createRagChain(st.session_state.vector_store)
        st.session_state.retriever = retriever
        st.session_state.rag_chain = chain
    
    if not st.session_state.messages:
        st.markdown("""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                height: 60vh;
                text-align: center;
            ">
                <h3 style="
                    font-size: 2rem;
                    color: #FFFFFF;
                    font-weight: 500;
                    opacity: 1;
                ">Hello! What can I assist you with today?</h3>
            </div>
        """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        renderChatMessage(
            message["role"],
            message["content"],
            message.get("sources"),
            message.get("is_error", False)
        )
    
    if prompt := st.chat_input("Ask about insurance coverage..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        renderChatMessage("user", prompt)
        
        with st.chat_message("assistant", avatar="📋"):
            with st.spinner("Analyzing policy documents..."):
                try:
                    sources = st.session_state.retriever.invoke(prompt)
                    
                    answer = st.session_state.rag_chain.invoke({
                        "question": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    st.markdown(answer)
                    
                    if sources:
                        uniqueSources = list(set([
                            doc.metadata.get("source", "Unknown") 
                            for doc in sources
                        ]))
                        if uniqueSources:
                            with st.expander("View Sources"):
                                for source in uniqueSources:
                                    st.caption(f"• {source}")
                    
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    st.rerun()
                    
                except Exception as e:
                    errorStr = str(e).lower()
                    
                    if "429" in errorStr or "rate limit" in errorStr:
                        errorMsg = "⚠️ Too many requests. Please try again in a moment."
                    elif any(code in errorStr for code in ["400", "401", "403", "404", "500", "502", "503", "504"]):
                        errorMsg = "⚠️ Unable to process your request. The AI service is currently unavailable."
                    elif "timeout" in errorStr or "timed out" in errorStr:
                        errorMsg = "⚠️ Request timed out. Please try again."
                    elif "api" in errorStr or "key" in errorStr:
                        errorMsg = "⚠️ API configuration error. Please check your settings."
                    else:
                        errorMsg = "⚠️ An unexpected error occurred. Please try again."
                    
                    st.markdown(f'<p style="color: #ff4b4b; font-weight: 500;">{errorMsg}</p>', unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": errorMsg,
                        "is_error": True
                    })
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

