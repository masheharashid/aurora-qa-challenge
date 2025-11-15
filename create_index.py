import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Path to your saved messages file
LOCAL_MESSAGES_FILE = "api_messages.json"

def load_messages_from_file():
    with open(LOCAL_MESSAGES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]

def build_documents(messages):
    """Combine user_name and message into a document string."""
    docs = []
    for m in messages:
        text = f"{m['user_name']}: {m['message']}"
        docs.append(text)
    return docs

def main():
    print("Loading local messages.json...")
    messages = load_messages_from_file()
    print(f"Loaded {len(messages)} messages.")

    print("Building documents...")
    docs = build_documents(messages)

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding messages...")
    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, "faiss_index.bin")

    # Save metadata
    print("Saving metadata...")
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

    print("FAISS index + metadata saved successfully!")

if __name__ == "__main__":
    main()
