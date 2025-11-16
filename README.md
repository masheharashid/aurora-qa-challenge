# Aurora AI/ML Engineer Challenge - Member QA System

A natural language question-answering system that answers questions about member data using semantic search and LLM-based extraction.

## Table of Contents
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Dependencies](#dependencies)
- [Bonus Goals](#bonus-goals)

## Live Demo
**API Endpoint:** [https://masheharashid-aurora-qa-challenge.hf.space/docs](https://masheharashid-aurora-qa-challenge.hf.space/docs)

**Try a sample question**: [https://masheharashid-aurora-qa-challenge.hf.space/ask?q=What is Vikram's office address?](https://masheharashid-aurora-qa-challenge.hf.space/ask?q=What%20is%20Vikram%27s%20office%20address?)
  
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

### HuggingFace Space Deployment 

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Click the **"Create new Space"** button and fill out the form
3. Click **"Create Space"**
4. Upload files directly or add HuggingFace as a remote and push
   - Make sure the application port is set to the right value (default: 7860 for HF, 8080 for local)
6. Click **"Commit changes to main"** or run the command ```git push hf master:main```
7. In your Space, click the **"Settings"** tab
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

## Bonus Goals

### Bonus 1: Design Notes

When building this question-answering system, I evaluated two different architectural approaches before settling on the final design. Here are the two alternatives I considered. 

#### Approach 1: **Pure LLM-based System** 

I considered sending each question directly to an LLM (like GPT-4 or Claude) with the entire message dataset included in the context window. The reasons I considered this approach are that it would be the simplest to implement, as it was just API calls and no machine learning infrastructure. LLMs also excel in understanding natural language nuances, so they could handle complex, multi-part questions naturally. 

I didn't choose this approach for the following reasons:
- **Cost concerns**: Sending 90+ messages in context for every query would be expensive at scale
- **Latency issues**: Processing large contexts is slow (about 3-5+ seconds per query)
- **Token limits**: Current dataset fits, but wouldn't scale beyond ~500-1000 messages

#### Approach 2: **Fine-Tuned Question-Answering Model** 

I considered fine-tuning a model like BERT, RoBERTa, or T5 specifically on member message data to create a custom QA system. The reasons I considered this approach are that there were no external API dependencies, so there are cost savings in the long term. Also, it would provide the best possible accuracy for this specific domain due having complete control over the model. 

I didn't choose this approach for the following reasons:
- **Time constraints**: Fine-tuning requires significant experimentation and iteration
- **Maintenance burden**: Model needs retraining whenever message patterns change
- **No training data**: Would need hundreds of labeled question-answer pairs
