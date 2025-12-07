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


def renderChatMessage(role: str, content: str, sources: List = None):
    """
    Render a chat message with optional source citations.
    
    Args:
        role: Message role ('user' or 'assistant').
        content: Message content.
        sources: Optional list of source documents.
    """
    with st.chat_message(role):
        st.markdown(content)
        
        if sources and role == "assistant":
            uniqueSources = list(set([doc.metadata.get("source", "Unknown") for doc in sources]))
            if uniqueSources:
                with st.expander("📄 View Sources"):
                    for source in uniqueSources:
                        st.caption(f"• {source}")


def main():
    """Main application entry point."""
    
    st.set_page_config(
        page_title="Insurance Policy Assistant",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stApp {
            max-width: 900px;
            margin: 0 auto;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏥 Insurance Policy Assistant")
    st.caption("Ask questions about UnitedHealthcare commercial medical policies")
    
    with st.sidebar:
        st.header("📄 Document Management")
        
        with st.expander("📥 Upload PDFs", expanded=True):
            st.markdown("""
            **How to add policy documents:**
            
            1. **Download PDFs manually** from the policy site
               - [UHC Policy Portal](https://www.uhcprovider.com/en/policies-protocols/commercial-policies/commercial-medical-drug-policies.html)
               - Open in Safari, accept Terms & Conditions
               - Download the PDFs you need
            
            2. **Upload them here** (supports multiple files)
            """)
            
            uploadedFiles = st.file_uploader(
                "Select PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader",
                help="You can select multiple PDFs at once (Cmd+Click)"
            )
            
            if uploadedFiles:
                if st.button("📥 Process Uploaded PDFs"):
                    processUploadedFiles(uploadedFiles)
                    st.session_state.pop("vector_store", None)
                    st.session_state.pop("rag_chain", None)
                    st.session_state.pop("retriever", None)
                    if VECTOR_STORE_PATH.exists():
                        shutil.rmtree(VECTOR_STORE_PATH)
                    st.success(f"✅ Uploaded {len(uploadedFiles)} PDF(s)")
                    st.rerun()
        
        with st.expander("🔗 Download from PDF URL", expanded=False):
            st.markdown("""
            **Have a direct PDF link?** Paste it here to download.
            
            Example: `https://example.com/policy.pdf`
            """)
            
            pdfUrl = st.text_input(
                "PDF URL",
                placeholder="https://example.com/document.pdf",
                key="pdf_url_input"
            )
            
            if st.button("📥 Download PDF"):
                if pdfUrl:
                    with st.spinner("Downloading..."):
                        downloadedFiles = downloadPolicies([pdfUrl], DATA_DIR)
                        if downloadedFiles:
                            st.session_state.pop("vector_store", None)
                            st.session_state.pop("rag_chain", None)
                            st.session_state.pop("retriever", None)
                            if VECTOR_STORE_PATH.exists():
                                shutil.rmtree(VECTOR_STORE_PATH)
                            st.success(f"✅ Downloaded: {downloadedFiles[0].name}")
                            st.rerun()
                        else:
                            st.error("Download failed. Check the URL and try again.")
                else:
                    st.warning("Please enter a URL")
        
        st.divider()
        
        existingPdfs = getExistingPdfs()
        if existingPdfs:
            st.markdown(f"### 📚 Loaded Documents ({len(existingPdfs)})")
            with st.container(height=200):
                for pdf in existingPdfs:
                    st.caption(f"📄 {pdf.name}")
        else:
            st.info("📂 No documents loaded yet")
        
        st.divider()
        
        st.header("💡 Example Questions")
        st.markdown("""
        - "Is bariatric surgery covered?"
        - "What are the requirements for gastric bypass?"
        - "Is Xolair covered for asthma?"
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.pop("rag_chain", None)
                st.session_state.pop("retriever", None)
                st.rerun()
        
        with col2:
            if st.button("🔄 Rebuild Index"):
                st.session_state.pop("vector_store", None)
                st.session_state.pop("rag_chain", None)
                st.session_state.pop("retriever", None)
                if VECTOR_STORE_PATH.exists():
                    shutil.rmtree(VECTOR_STORE_PATH)
                st.rerun()
    
    existingPdfs = getExistingPdfs()
    
    if not existingPdfs:
        st.info("👈 Upload policy PDFs using the sidebar to get started.")
        st.markdown("""
        ### 📋 Quick Start Guide
        
        1. **Visit** the [UHC Policy Portal](https://www.uhcprovider.com/en/policies-protocols/commercial-policies/commercial-medical-drug-policies.html) in Safari
        2. **Accept** Terms & Conditions
        3. **Download** the policy PDFs you need (Cmd+Click for multiple)
        4. **Upload** them using the sidebar →
        5. **Ask questions** once loaded!
        """)
        st.stop()
    
    if "vector_store" not in st.session_state:
        vectorStore, embeddings, chunkCount = initializePipelineFromPdfs(existingPdfs)
        if vectorStore is None:
            st.stop()
        st.session_state.vector_store = vectorStore
        st.session_state.chunk_count = chunkCount
    
    st.success(f"✅ Loaded {st.session_state.chunk_count} document chunks from {len(existingPdfs)} policy document(s)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "rag_chain" not in st.session_state:
        retriever, chain = createRagChain(st.session_state.vector_store)
        st.session_state.retriever = retriever
        st.session_state.rag_chain = chain
    
    for message in st.session_state.messages:
        renderChatMessage(
            message["role"],
            message["content"],
            message.get("sources")
        )
    
    if prompt := st.chat_input("Ask about insurance coverage..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        renderChatMessage("user", prompt)
        
        with st.chat_message("assistant"):
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
                            with st.expander("📄 View Sources"):
                                for source in uniqueSources:
                                    st.caption(f"• {source}")
                    
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    errorMsg = f"Error generating response: {str(e)}"
                    st.error(errorMsg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": errorMsg
                    })


if __name__ == "__main__":
    main()

