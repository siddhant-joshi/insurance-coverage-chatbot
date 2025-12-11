# Insurance Policy Assistant

An AI-powered chatbot that helps you understand UnitedHealthcare commercial medical policies. Built with Streamlit and powered by Google's Gemini 2.0 Flash, this application uses advanced RAG (Retrieval-Augmented Generation) technology to provide accurate answers based on actual policy documents.

## What It Does

This chatbot makes it easy to find answers in insurance policy documents without having to manually search through hundreds of pages of PDFs. Simply upload your policy documents, ask questions in plain English, and get accurate answers with source citations showing exactly where the information came from.

The application includes features like document management, conversation memory for follow-up questions, and a clean interface optimized for professional use.

## Key Features

**Document Management**
Upload multiple policy PDFs through the sidebar, view all loaded documents at a glance, and delete documents you no longer need. The system automatically processes and indexes your documents for quick retrieval.

**Intelligent Search**
Uses semantic search technology (FAISS vector store with sentence-transformers) to find the most relevant sections of your policy documents, even when questions are phrased differently than the document text.

**AI-Powered Answers**
Leverages Google's Gemini 2.0 Flash model to generate natural, easy-to-understand answers based solely on your uploaded policy documents.

**Source Citations**
Every answer includes references to the specific documents used, so you can verify the information and dive deeper if needed.

**Conversation Memory**
The chatbot remembers your conversation, making it easy to ask follow-up questions without repeating context.

**Professional Interface**
Full-width layout optimized for large screens, with a fixed sidebar for document management and a scrollable document list that stays organized even with many files.

## Technology Stack

The application is built on a modern AI stack:

- **Frontend:** Streamlit provides the web interface
- **RAG Framework:** LangChain handles the retrieval and response generation pipeline
- **Vector Database:** FAISS stores and searches document embeddings efficiently
- **Embeddings:** HuggingFace's sentence-transformers/all-MiniLM-L6-v2 model converts text into searchable vectors
- **Language Model:** Google Gemini 2.0 Flash generates the final answers
- **PDF Processing:** PyPDF extracts text from policy documents

Everything runs locally except for the API calls to Google's Gemini model.

## Getting Started

### What You'll Need

- Python 3.8 or newer
- A Google API key (free to get, see below)
- About 5 minutes for setup

### Installation

First, make sure you're in the project directory:

```bash
cd insurance-coverage-chatbot
```

Install all the required packages:

```bash
pip install -r requirements.txt
```

This will install Streamlit, LangChain, FAISS, and all other dependencies automatically.

### Setting Up Your API Key

You'll need a Google API key to use the Gemini language model. Don't worry, it's free and takes just a minute to set up.

1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Create a file named `.env` in the project root
4. Add this line to the file: `GOOGLE_API_KEY=your_actual_api_key_here`

Replace `your_actual_api_key_here` with the key you just created.

### Running the Application

Start the application with:

```bash
streamlit run app.py
```

Your default web browser will automatically open to `http://localhost:8501` with the application running.

## How to Use

### Loading Your First Documents

When you first open the application, you'll need to upload some policy documents:

1. Look at the sidebar on the left
2. Click on the "Upload PDFs" section
3. You'll see instructions for downloading PDFs from the UHC Policy Portal
4. Use the file selector to choose one or more PDF files
5. Click "Process Uploaded PDFs"

The application will process your documents (this takes a bit longer the first time) and create a searchable index. This index is saved locally, so subsequent runs will be much faster.

### Asking Questions

Once your documents are loaded, just type your question in the chat box at the bottom of the screen. Here are some examples:

For coverage questions:
- "Is bariatric surgery covered?"
- "Does the policy cover Xolair for asthma?"

For requirements and criteria:
- "What are the requirements for gastric bypass surgery?"
- "What conditions need to be met for weight loss surgery approval?"

For exclusions and limitations:
- "What are the exclusions for bariatric surgery?"
- "Are there age restrictions for this procedure?"

For procedural questions:
- "What documentation is needed for prior authorization?"
- "How long is the approval valid?"

The chatbot will search through your documents and provide an answer based on what it finds. Click "View Sources" under any answer to see which documents were referenced.

### Managing Documents

**Viewing Documents**
All loaded documents appear in the sidebar with their filenames. The count is shown at the top of the list.

**Removing Documents**
Click the "Delete" button next to any document to remove it. The search index will automatically update.

**Rebuilding the Index**
If something seems off or you want to start fresh, click "Rebuild Index" in the sidebar. This recreates the entire search index from scratch.

**Clearing Chat History**
Use the "Clear Chat" button to reset the conversation while keeping all your documents loaded.

## Project Organization

Here's what's in the project folder:

```
insurance-coverage-chatbot/
├── app.py                 # The main application code
├── requirements.txt       # List of Python packages needed
├── .env                   # Your API key (you create this)
├── .gitignore            # Tells git what not to track
├── README.md             # This file
└── data/                 # Created automatically when you run the app
    ├── *.pdf             # Your uploaded policy documents
    └── vector_store/     # The search index (cached for speed)
        ├── index.faiss
        └── index.pkl
```

## Configuration Details

The application uses sensible defaults, but here's what's happening under the hood:

**Document Processing**
Documents are split into chunks of 1000 characters with 200 characters of overlap to maintain context. The splitting happens at paragraph breaks, line breaks, and spaces (in that order of preference).

**Search Settings**
When you ask a question, the system retrieves the 4 most relevant document chunks using similarity search. These chunks are sent to the language model along with your question.

**Model Settings**
The Gemini model is set to a temperature of 0.3, which means it stays fairly focused and factual rather than being creative. The embeddings use the sentence-transformers/all-MiniLM-L6-v2 model, which is fast and accurate for this use case.

## Behind the Scenes

### Caching for Speed
After the first run, the search index is saved to `data/vector_store/`. This means subsequent runs are much faster because the documents don't need to be reprocessed. The cache is automatically invalidated when you add or remove documents.

### How Conversation Works
The chatbot uses LangChain's conversation memory to keep track of what you've asked before. This allows for natural follow-up questions like "What about the age requirements?" after asking about coverage.

### Error Handling
The application handles various error scenarios gracefully:
- If a PDF can't be processed, it shows an error but continues with other documents
- Missing API keys trigger a clear error message
- Network issues during document downloads include retry logic
- Failed searches return helpful error messages rather than cryptic stack traces

## A Few Notes

**Performance:** The first time you load documents takes longer because the system needs to generate embeddings. After that, it's much faster thanks to caching.

**Storage:** The search index is stored locally in the `data/` directory. Make sure you have a few hundred MB of free space if you're loading many large documents.

**Privacy:** All document processing happens on your computer. The only things sent over the internet are the relevant excerpts and your questions to Google's Gemini API.

**Accuracy:** Answers are limited to what's in your uploaded documents. If information isn't in the PDFs, the chatbot will tell you rather than making something up.

**Cost:** Using Gemini is very inexpensive. The free tier includes 1,500 requests per day, and beyond that, each question costs roughly $0.0001.

## Troubleshooting

**"GOOGLE_API_KEY not found"**
This means the application can't find your API key. Double-check that you created a `.env` file in the project root (not in a subfolder) and that it contains the line `GOOGLE_API_KEY=your_key_here` with no extra spaces or quotes.

**"No documents loaded yet"**
You need to upload at least one PDF before you can ask questions. Use the file uploader in the sidebar to add documents.

**Vector Store Errors**
If you see errors about the vector store or index, try clicking "Rebuild Index" in the sidebar. If that doesn't work, you can manually delete the `data/vector_store/` folder and restart the application.

**Slow Responses**
The first query in a session is always slower because the embedding model needs to initialize. After that, responses should be quick (usually under 5 seconds). If everything is slow, check your internet connection since the LLM queries happen via API.

**Documents Not Processing**
Make sure your PDFs are valid and not password-protected. The application uses PyPDF for extraction, which works with most standard PDFs but may struggle with scanned documents that haven't been OCR'd.

## Support

If you run into issues:

1. Check the troubleshooting section above
2. Look at the console output where you ran `streamlit run app.py` - it often has detailed error messages
3. Make sure all dependencies are installed correctly
4. Verify your API key is valid and has available quota

The application shows detailed error messages in the interface when something goes wrong, so read those carefully - they usually point to the solution.

---

Built with Streamlit, LangChain, and Google Gemini AI
