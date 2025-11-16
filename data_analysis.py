import json
from collections import Counter, defaultdict
import re
from datetime import datetime

# Load data
with open("metadata.json", "r") as f:
    messages = json.load(f)

print("="*80)
print("DATASET ANALYSIS REPORT")
print("="*80)

# Basic stats
print(f"\n BASIC STATISTICS")
print(f"Total messages: {len(messages)}")
users = set(msg['user_name'] for msg in messages)
print(f"Total unique users: {len(users)}")
print(f"Average messages per user: {len(messages)/len(users):.1f}")

# Date range
timestamps = [datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00')) for msg in messages]
print(f"Date range: {min(timestamps).strftime('%Y-%m-%d')} to {max(timestamps).strftime('%Y-%m-%d')}")

# Messages per user
print(f"\n MESSAGES PER USER")
user_counts = Counter(msg['user_name'] for msg in messages)
for user, count in sorted(user_counts.items(), key=lambda x: -x[1]):
    print(f"  {user}: {count} messages")

# Encoding issues
print(f"\n ENCODING ISSUES DETECTED")
encoding_issues = [msg for msg in messages if any(char in msg['message'] for char in ['â€"', 'â€™', 'â€œ', 'Ã'])]
print(f"Messages with encoding problems: {len(encoding_issues)}")
if encoding_issues:
    print("Examples:")
    for msg in encoding_issues[:3]:
        snippet = msg['message'][:80] + "..." if len(msg['message']) > 80 else msg['message']
        print(f"  - {snippet}")

# Topic distribution (keyword-based estimation)
print(f"\n TOPIC DISTRIBUTION (Estimated)")
topics = {
    'Travel': ['travel', 'trip', 'flight', 'fly', 'jet', 'hotel', 'villa', 'book'],
    'Restaurants': ['restaurant', 'dinner', 'table', 'reservation', 'lunch'],
    'Account': ['update', 'profile', 'contact', 'number', 'address', 'card'],
    'Service': ['thank', 'help', 'assist', 'issue', 'problem', 'confirm']
}

topic_counts = defaultdict(int)
for msg in messages:
    msg_lower = msg['message'].lower()
    for topic, keywords in topics.items():
        if any(kw in msg_lower for kw in keywords):
            topic_counts[topic] += 1

total_categorized = sum(topic_counts.values())
for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
    percentage = (count/len(messages))*100
    print(f"  {topic}: {count} messages ({percentage:.1f}%)")

# Date mentions
print(f"\n DATE PATTERNS")
absolute_dates = sum(1 for msg in messages if re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}', msg['message'], re.IGNORECASE))
relative_dates = sum(1 for msg in messages if re.search(r'\b(this|next|tomorrow|today|tonight)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week)', msg['message'], re.IGNORECASE))
print(f"  Absolute dates (e.g., 'November 15'): {absolute_dates} messages")
print(f"  Relative dates (e.g., 'this Friday'): {relative_dates} messages")

# Number mentions
print(f"\n NUMERICAL DATA")
numbers_with_context = []
for msg in messages:
    matches = re.findall(r'\b(\d+)\s+(\w+)', msg['message'])
    numbers_with_context.extend(matches)

if numbers_with_context:
    context_counts = Counter(match[1] for match in numbers_with_context)
    print(f"  Total number mentions: {len(numbers_with_context)}")
    print(f"  Most common contexts:")
    for context, count in context_counts.most_common(5):
        print(f"    - {context}: {count}")

# Restaurant extraction
print(f"\n RESTAURANT MENTIONS")
restaurant_pattern = r'\bat\s+([A-Z][A-Za-z\s&\'-]{2,30}?)(?:\s+(?:for|on|tonight|tomorrow|this|next)|\.|,|$)'
restaurants = []
for msg in messages:
    matches = re.findall(restaurant_pattern, msg['message'])
    restaurants.extend(matches)

unique_restaurants = set(r.strip() for r in restaurants)
print(f"  Total restaurant mentions: {len(restaurants)}")
print(f"  Unique restaurants: {len(unique_restaurants)}")
if unique_restaurants:
    print(f"  Examples: {', '.join(list(unique_restaurants)[:5])}")

# False positives
print(f"\n POTENTIAL FALSE POSITIVES")
car_mentions = [msg for msg in messages if 'car' in msg['message'].lower()]
actual_cars = [msg for msg in car_mentions if any(w in msg['message'].lower() for w in ['vehicle', 'garage', 'drive'])]
false_positives = len(car_mentions) - len(actual_cars)
print(f"  'car' mentions: {len(car_mentions)} (false positives: ~{false_positives})")
print(f"  Examples of false positives:")
for msg in car_mentions[:2]:
    if 'card' in msg['message'].lower():
        snippet = msg['message'][:70] + "..."
        print(f"    - {snippet}")

# Data gaps
print(f"\n DATA GAPS")
number_pattern = r'\b\d+\b'
messages_with_numbers = len([m for m in messages if re.search(number_pattern, m['message'])])
car_vehicle_keywords = ['car', 'vehicle', 'auto']
messages_with_cars = len([m for m in messages if any(w in m['message'].lower() for w in car_vehicle_keywords)])
print(f"  Messages with explicit numbers: {messages_with_numbers}")
print(f"  Messages with car/vehicle mentions: {messages_with_cars}")
print(f"  Messages with dates: {absolute_dates + relative_dates}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)