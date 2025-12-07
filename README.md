# 🏥 Insurance Policy RAG Chatbot

A Proof of Concept (POC) RAG-based chatbot that answers questions about insurance coverage based on UnitedHealthcare's commercial medical policies.

## Features

- **PDF Document Ingestion**: Automatically downloads and processes policy documents
- **Semantic Search**: Uses FAISS vector store with sentence-transformers for accurate retrieval
- **AI-Powered Responses**: Leverages Google Gemini for natural language understanding
- **Source Citations**: Shows which policy documents were used to answer questions
- **Chat History**: Maintains conversation context for follow-up questions

## Tech Stack

- **UI**: Streamlit
- **RAG Framework**: LangChain
- **Vector Store**: FAISS (CPU)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM**: Google Gemini Flash

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

On first run, the application will:
1. Download policy PDFs to the `data/` folder
2. Process and chunk the documents
3. Create vector embeddings (cached for subsequent runs)

Then you can ask questions like:
- "Is bariatric surgery covered?"
- "What are the requirements for gastric bypass surgery?"
- "Is Xolair covered for asthma treatment?"
- "What conditions must be met for weight loss surgery approval?"

## Available Policies

The POC includes these UnitedHealthcare policies:
- Bariatric Surgery Commercial Medical Policy
- Xolair Prior Authorization Requirements

## Project Structure

```
chatbot/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .env                # Your API key (create this)
├── data/               # Downloaded PDFs & vector store
└── README.md           # This file
```

## Notes

- The vector store is cached locally after first creation
- Clear chat history using the sidebar button to reset the conversation
- Source documents are shown in an expandable section under each response

