from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
import uvicorn
from typing import List, Dict, Optional
import os
import requests  
import dateparser  
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()

app = FastAPI()

# ---------------------------
# Configuration
# ---------------------------

# Use OpenRouter API (gpt-oss-20b) 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  
USE_LLM = OPENROUTER_API_KEY is not None

print(f"LLM-based extraction: {'ENABLED' if USE_LLM else 'DISABLED'}")

# ---------------------------
# Load FAISS + metadata
# ---------------------------

print("Loading MiniLM model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("faiss_index.bin")

print("Loading metadata...")
with open("metadata.json", "r") as f:
    metadata = json.load(f)


# ---------------------------
# Helper functions
# ---------------------------

def extract_person_name(question: str) -> Optional[str]:
    """Extract person's name from the question."""
    patterns = [
        r"(?:is|does|did|has|have|are)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)*?)(?:'s|\s+)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)*?)(?:'s|')",
        r"about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)*)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            name = match.group(1).strip()
            if name.lower() not in ['when', 'what', 'where', 'who', 'how', 'why', 'many']:
                return name
    return None


def filter_by_person(docs: List[Dict], person_name: str) -> List[Dict]:
    if not person_name:
        return docs
    person_lower = person_name.lower()
    filtered = []
    for doc in docs:
        user_name_lower = doc['user_name'].lower()
        name_parts = person_lower.split()
        if any(part in user_name_lower for part in name_parts):
            filtered.append(doc)
    return filtered if filtered else docs


def keyword_filter(docs: List[Dict], keywords: List[str], require_any: bool = True) -> List[Dict]:
    filtered = []
    for doc in docs:
        msg_lower = doc['message'].lower()
        if require_any:
            if any(kw in msg_lower for kw in keywords):
                filtered.append(doc)
        else:
            if all(kw in msg_lower for kw in keywords):
                filtered.append(doc)
    return filtered


def semantic_search(query: str, k: int = 20) -> List[Dict]:
    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, k)
    results = [metadata[i] for i in indices[0]]
    return results


def normalize_date(text: str) -> Optional[str]:
    """Normalize relative or absolute date to YYYY-MM-DD format."""
    dt = dateparser.parse(text, settings={'RELATIVE_BASE': datetime.now()})
    if dt:
        return dt.strftime("%Y-%m-%d")
    return None

def extract_number(text: str, subject_keywords: List[str]) -> Optional[int]:
    """Extract number associated with subjects like 'cars', 'tickets', etc."""
    for kw in subject_keywords:
        pattern = rf'(\d+)\s*(?:{kw})|(?:{kw})\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1) or match.group(2))
    return None


def llm_extract_answer(question: str, messages: List[Dict]) -> Optional[str]:
    """Use OpenRouter gpt-oss-20b for RAG-based extraction."""
    if not USE_LLM or not messages:
        return None

    # Format messages (RAG: provide top relevant messages)
    context = "\n\n".join([
        f"Message from {msg['user_name']} on {msg['timestamp'][:10]}:\n{msg['message']}"
        for msg in messages[:5]  # top 5 docs to save tokens
    ])

    prompt = f"""Based on the following messages, answer this question: "{question}"

Messages:
{context}

Instructions:
- If the answer is a date, return it in YYYY-MM-DD format
- If the answer is a number, return just the number
- If the answer is a list (like restaurants), return a JSON array
- If you cannot find the answer, return "UNABLE_TO_ANSWER"
- Be concise and only return the direct answer, nothing else

Answer:"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 300
            },
            timeout=30
        )
        data = response.json()
        answer = data['choices'][0]['message']['content'].strip()

        # Parse JSON if it looks like a list
        if answer.startswith("[") and answer.endswith("]"):
            return json.loads(answer)

        return answer if answer != "UNABLE_TO_ANSWER" else None

    except Exception as e:
        print(f"LLM extraction error: {e}")
        return None


def rule_based_extract(question: str, docs: List[Dict]) -> Optional[str]:
    q_lower = question.lower()
    
    # DATE QUESTIONS
    if any(word in q_lower for word in ["when", "date"]):
        date_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b'
        relative_patterns = [
            r'\bthis (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\bnext (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\btomorrow\b',
            r'\btoday\b'
        ]
        for doc in docs:
            # Absolute dates
            match = re.search(date_pattern, doc['message'], re.IGNORECASE)
            if match:
                normalized = normalize_date(match.group(0))
                if normalized:
                    return normalized
            # Relative dates
            for pattern in relative_patterns:
                match = re.search(pattern, doc['message'], re.IGNORECASE)
                if match:
                    normalized = normalize_date(match.group(0))
                    if normalized:
                        return normalized

    # NUMBER QUESTIONS
    if "how many" in q_lower:
        subjects = ["car", "cars", "ticket", "tickets", "people", "guests", "rooms"]
        numbers = []
        for doc in docs:
            num = extract_number(doc['message'], subjects)
            if num is not None:
                numbers.append(num)
        if numbers:
            return numbers[0]  # Return first found number

    # RESTAURANT QUESTIONS
    if "restaurant" in q_lower or "favorite" in q_lower:
        restaurants = []
        for doc in docs:
            patterns = [
                r'\bat\s+([A-Z][A-Za-z\s&\'-]{2,30}?)(?:\s+(?:for|on|tonight|tomorrow|this|next)|\.|,|$)',
                r'\breserve.*?at\s+([A-Z][A-Za-z\s&\'-]{2,30}?)(?:\s+(?:for|on)|\.|,|$)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, doc['message'])
                for match in matches:
                    name = match.strip()
                    name = re.sub(r'\s+(for|on|at|in|the)$', '', name, flags=re.IGNORECASE)
                    if len(name) > 2:
                        restaurants.append(name)
        if restaurants:
            return list(dict.fromkeys(restaurants))

    # FALLBACK: UNABLE TO ANSWER
    return None  

# ---------------------------
# Main QA endpoint
# ---------------------------

@app.get("/ask")
def ask(q: str = Query(...)):
    person_name = extract_person_name(q)
    all_docs = semantic_search(q, k=50)
    person_docs = filter_by_person(all_docs, person_name) if person_name else all_docs

    if not person_docs:
        return {"answer": "UNABLE_TO_ANSWER"}

    q_lower = q.lower()

    # --- Initial keyword filtering ---
    filtered_docs = person_docs

    # Travel-related questions
    if any(word in q_lower for word in ["travel", "trip", "fly", "flight", "paris", "visit"]):
        keywords = ["trip", "travel", "flight", "fly", "book", "jet", "visit", "itinerary", "paris"]
        filtered_docs = keyword_filter(person_docs, keywords)

    # Car count questions
    elif "how many" in q_lower and "car" in q_lower:
        keywords = ["car", "vehicle", "auto", "garage"]
        filtered_docs = keyword_filter(person_docs, keywords)

    # Restaurant questions
    elif "restaurant" in q_lower or "favorite" in q_lower:
        keywords = ["restaurant", "dinner", "table", "reserve", "reservation", "lunch", "eat"]
        filtered_docs = keyword_filter(person_docs, keywords)

    # If the question is about WHEN something happens, do NOT restrict by keywords
    if "when" in q_lower:
        filtered_docs = person_docs

    # Fallback if keyword filtering removed everything
    final_docs = filtered_docs if filtered_docs else person_docs

    # LLM extraction
    if USE_LLM:
        answer = llm_extract_answer(q, final_docs)
        if answer:
            return {"answer": answer}

    # Rule-based extraction
    answer = rule_based_extract(q, final_docs)
    if answer:
        return {"answer": answer}

    return {"answer": "No relevant information found."}


@app.get("/")
def root():
    return {
        "service": "Member QA System",
        "endpoint": "/ask?q=YOUR_QUESTION",
        "llm_enabled": USE_LLM
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)