import requests
import json

API_URL = "https://november7-730026606190.europe-west1.run.app/messages"

def fetch_messages(save_path="api_messages.json"):
    print("Fetching messages...")
    response = requests.get(API_URL)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    data = response.json()

    # Optional: Save the entire dataset locally
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {data['total']} messages to {save_path}")
    return data

# Example run
if __name__ == "__main__":
    fetch_messages()
