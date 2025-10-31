# RAG Demo: State of the Union Speech Analysis

A demonstration comparing Retrieval Augmented Generation (RAG) vs vanilla LLM responses using Anthropic's Claude API and the classic "State of the Union" speech.

## 🎯 What This Demonstrates

This demo shows how RAG improves LLM responses by:
- **Vanilla LLM**: Claude answers based purely on its training data (may be outdated or lack specific details)
- **RAG-Enhanced**: Claude answers using retrieved, relevant excerpts from the actual speech (accurate, specific, grounded)

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- An Anthropic API key ([get one here](https://console.anthropic.com/))

### Quick Setup (Recommended)

1. **Run the setup script:**
   ```bash
   bash setup.sh
   ```

2. **Set your API key:**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

3. **Run the demo:**
   ```bash
   source venv/bin/activate  # If not already activated
   python rag_demo.py
   ```

### Manual Setup

If you prefer to set up manually:

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set your API key:**
   ```bash
   # On Linux/Mac:
   export ANTHROPIC_API_KEY='your-api-key-here'
   
   # On Windows (Command Prompt):
   set ANTHROPIC_API_KEY=your-api-key-here
   
   # On Windows (PowerShell):
   $env:ANTHROPIC_API_KEY="your-api-key-here"
   ```

5. **Run the demo:**
   ```bash
   python rag_demo.py
   ```

## 🔑 Getting Your Anthropic API Key

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and use it in the setup steps above

## 📋 What the Demo Does

1. **Downloads** the 2022 State of the Union speech
2. **Chunks** the text into manageable pieces
3. **Creates embeddings** using sentence-transformers
4. **Compares responses** for questions like:
   - "What did Biden say about Ukraine?"
   - "What did Biden say about inflation?"
   - "What did Biden mention about COVID-19?"

For each question, you'll see:
- ❌ Vanilla LLM response (without context)
- ✅ RAG-enhanced response (with retrieved context)
- 📚 The actual chunks retrieved from the speech

## 🧩 How It Works

```
User Question: "What did Biden say about Ukraine?"
        ↓
    [Embedding Model]
        ↓
    Query Embedding
        ↓
    [Similarity Search]
        ↓
    Top 3 Relevant Chunks
        ↓
    [LLM with Context]
        ↓
    Accurate, Grounded Answer
```

## 🔧 Architecture

- **Embedding Model**: `all-MiniLM-L6-v2` (fast, efficient sentence embeddings)
- **LLM**: Claude Sonnet 4 via Anthropic API
- **Retrieval**: Cosine similarity between query and chunk embeddings
- **Chunking**: 500 words per chunk with 50-word overlap

## 📦 Dependencies

- `anthropic` - Anthropic API client
- `sentence-transformers` - For creating embeddings
- `scikit-learn` - For similarity calculations
- `numpy` - Numerical operations
- `torch` - PyTorch (required by sentence-transformers)

## 💡 Customization

You can modify the demo to:
- Use different documents (change the `load_state_of_union` method)
- Adjust chunk size and overlap
- Change the number of retrieved chunks (top_k)
- Try different embedding models
- Add your own questions

## 🧪 Example Output

```
QUESTION: What did Biden say about Ukraine?
================================================================================

📝 VANILLA LLM RESPONSE (No Context):
--------------------------------------------------------------------------------
I don't have access to specific State of the Union addresses...

🔍 RAG-ENHANCED RESPONSE (With Retrieved Context):
--------------------------------------------------------------------------------
Biden spoke about Ukraine's resistance against Russian invasion, praising 
the courage of President Zelenskyy and the Ukrainian people...

📚 RETRIEVED CONTEXT (Top 3 chunks):
--------------------------------------------------------------------------------
Chunk 1 (Similarity: 0.8234):
Six days ago, Russia's Vladimir Putin sought to shake the foundations...
```

## 🚀 Next Steps

After running this demo, you can:
- Experiment with your own documents
- Try different questions
- Adjust the RAG parameters (chunk size, top_k, etc.)
- Explore more advanced RAG techniques (hybrid search, reranking, etc.)

## 📝 Notes

- The first run will download the embedding model (~80MB)
- API calls to Anthropic will incur costs (check their pricing)
- The demo uses a fallback text if the download fails

## 🤝 Contributing

Feel free to modify and extend this demo for your own use cases!

## 📄 License

This is a demonstration project. Use freely for learning and experimentation.
