# Aurora AI/ML Engineer Challenge - Member QA System

A natural language question-answering system that answers questions about member data using semantic search and LLM-based extraction.

## Table of Contents
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#file-structure)
- [Deployment](#deployment)
- [Dependencies](#dependencies)
- [Bonus Goals](#bonus-goals)

## Live Demo
**API Endpoint:** [https://masheharashid-aurora-qa-challenge.hf.space/docs](https://masheharashid-aurora-qa-challenge.hf.space/docs)
  
## Architecture

### Components
  
1. **Semantic Search**: Uses `sentence-transformers` (all-MiniLM-L6-v2) to encode questions and find relevant messages
2. **Facebook AI Similarity Search (FAISS) Index**: Vector similarity search for efficient message retrieval
3. **LLM Extraction**: OpenAI's gpt-oss-20b for intelligent answer extraction
4. **Rule-Based Fallback**: Regular expression (regex) patterns for reliable extraction when LLM is unavailable

### Technology Stack

- **Framework**: FastAPI
- **ML Models**: Sentence Transformers, FAISS
- **LLM**: OpenAI gpt-oss-20b (Free API key from [OpenRouter](https://openrouter.ai/openai/gpt-oss-20b:free))
- **Deployment**: HuggingFace Spaces (Docker)
- **Language**: Python 3.10

## How It Works 

1. **Question Processing**: Extract person names and question type from natural language query
2. **Semantic Search**: Encode question and retrieve top-k relevant messages from FAISS index
3. **Filtering**: Apply person-based and keyword-based filtering to narrow results
4. **Answer Extraction**: 
   - Primary: Use LLM to intelligently extract answer from relevant messages
   - Fallback: Use rule-based patterns (regex) for reliable extraction
5. **Response**: Return structured JSON response with answer

## Project Structure
```
aurora-qa-challenge/
├── app.py                    # Main FastAPI application
├── create_index.py           # Script to build FAISS index
├── extract_responses.py      # Script to extract API responses
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration for HuggingFace Deployment
├── faiss_index.bin           # Pre-built FAISS index
└── metadata.json             # Message metadata
```

## Deployment 

### Local Deployment 

#### Prerequisites

- Python 3.10+
- pip

#### Setup
```bash
# Clone repository
git clone https://github.com/masheharashid/aurora-qa-challenge.git
cd aurora-qa-challenge

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On macOS & Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export OPENROUTER_API_KEY=your_api_key_here

# Extract API responses
python extract_responses.py

# Build FAISS index 
python build_index.py

# Run application
python app.py
```

The API will be available at `http://localhost:8080`

### Hugging Face Space Deployment 

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"** button and fill out the form
3. Click **"Create Space"**
4. Upload files directly or add HuggingFace as a remote and push
   - Make sure the application port is set to the right value (default: 7860 for HF, 8080 for local)
6. Click **"Commit changes to main"** or run the command ```git push hf master:main```
7. In your Space, click **"Settings"** tab
8. Scroll to **"Variables and secrets"**
9. Click **"New secret"** to add LLM API key
10. Add:
    - **Name**: `OPENROUTER_API_KEY`
    - **Value**: `your-api-key-here`
11. Go to the **"App"** tab and wait for the app to build
12. Once it says "Running", your API is live at: ```https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space```

## Dependencies

See `requirements.txt` for the full list. Key dependencies:

- `fastapi` - Web framework
- `sentence-transformers` - Semantic embeddings
- `faiss-cpu` - Vector similarity search
- `dateparser` - Date parsing and normalization
- `requests` - HTTP client for LLM API calls
