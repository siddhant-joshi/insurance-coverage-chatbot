# Insurance Policy Assistant - Final Version

## ✅ What Changed

We removed bot-protected web scraping features and focused on a reliable manual workflow that works with UHC's protected site.

---

## 🚀 How to Use

### 1. Run the App

```bash
cd /Users/ragaai_user/Desktop/chatbot/insurance-coverage-chatbot
source ../venv/bin/activate
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

### 2. Add Documents (Manual Workflow)

**Option A: Upload Local PDFs (Recommended)**

1. Visit [UHC Policy Portal](https://www.uhcprovider.com/en/policies-protocols/commercial-policies/commercial-medical-drug-policies.html) in **Safari**
2. Accept Terms & Conditions
3. Download policy PDFs you need (Cmd+Click for multiple)
4. In the app sidebar: Click "📥 Upload PDFs"
5. Select multiple PDFs at once
6. Click "Process Uploaded PDFs"

**Option B: Direct PDF URL**

If you have a direct PDF URL:
1. Expand "🔗 Download from Direct PDF URL"
2. Paste URL: `https://example.com/policy.pdf`
3. Click "Download PDF"

---

### 3. Ask Questions

Once PDFs are loaded:
- Type questions in the chat box at the bottom
- Examples:
  - "Is bariatric surgery covered?"
  - "What are the requirements for gastric bypass?"
  - "Is Xolair covered for asthma?"

---

## 🗂️ What Was Removed

To simplify and avoid bot-blocking issues:

- ❌ Web page text scraping
- ❌ Automated PDF discovery from portals
- ❌ Deep crawling features

These features don't work with UHC's site because:
- Bot protection (blocks Python requests)
- Terms & Conditions gate
- Browser fingerprinting (only Safari works)

---

## 📁 Project Structure

```
insurance-coverage-chatbot/
├── app.py                 # Main Streamlit app
├── data/                  # PDF storage
│   ├── *.pdf             # Uploaded/downloaded PDFs
│   └── vector_store/     # FAISS embeddings
├── .env                   # GOOGLE_API_KEY
└── .gitignore
```

---

## 🔑 Requirements

1. **Python 3.13** with virtual environment
2. **Google API Key** in `.env`:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
3. **Dependencies** (already installed in venv):
   - streamlit
   - langchain
   - langchain-google-genai
   - langchain-huggingface
   - FAISS
   - sentence-transformers

---

## 💰 Cost

- **Document processing**: FREE (local embeddings)
- **Asking questions**: Gemini API
  - Free tier: 1,500 requests/day
  - After: ~$0.0001 per question

---

## 🐛 Troubleshooting

**"No documents loaded"**
- Upload PDFs using sidebar
- Check `data/` folder has PDFs

**"GOOGLE_API_KEY not found"**
- Create `.env` file
- Add: `GOOGLE_API_KEY=your_key`
- Get key from: https://aistudio.google.com/app/apikey

**Vector store issues**
- Click "🔄 Rebuild Index" in sidebar
- Or delete `data/vector_store/` folder

---

## 🎯 Next Steps

The app is production-ready for:
- ✅ Uploading any number of PDFs
- ✅ Asking questions about policy content
- ✅ Tracking source documents
- ✅ Chat history within a session

Enjoy! 🎉

