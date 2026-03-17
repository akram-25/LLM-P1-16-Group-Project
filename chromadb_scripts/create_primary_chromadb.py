"""
Repopulate the original 'foodkaki_restaurants' collection
using food_places_primary.csv data.

Usage: python create_primary_chromadb.py
"""

import os
import csv
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ==========================================
# 1. LOAD CONFIG
# ==========================================
load_dotenv('apikeys.env')

CHROMA_SERVER_URL = os.getenv('CHROMA_SERVER_URL')
CHROMA_API_TOKEN = os.getenv('CHROMA_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

CSV_FILE = "food_places_primary.csv"
COLLECTION_NAME = "foodkaki_restaurants"
BATCH_SIZE = 50

# ==========================================
# 2. CONNECT TO CLOUD CHROMADB
# ==========================================
print("🔗 Connecting to Cloud ChromaDB...")

chroma_auth_client = chromadb.HttpClient(
    host=CHROMA_SERVER_URL,
    port=443,
    ssl=True,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials=CHROMA_API_TOKEN,
        anonymized_telemetry=False
    )
)

chroma_auth_client.heartbeat()
print("   ✅ Connected to ChromaDB server.")

existing = [c.name for c in chroma_auth_client.list_collections()]
print(f"   📦 Existing collections: {existing}")

if COLLECTION_NAME in existing:
    col = chroma_auth_client.get_collection(COLLECTION_NAME)
    count = col.count()
    if count > 0:
        print(f"   ⚠️  Collection '{COLLECTION_NAME}' has {count} docs. Deleting and recreating...")
    else:
        print(f"   ⚠️  Collection '{COLLECTION_NAME}' exists but is empty. Deleting and recreating...")
    chroma_auth_client.delete_collection(COLLECTION_NAME)
    print(f"   🗑️  Deleted old collection.")

# ==========================================
# 3. SETUP EMBEDDING FUNCTION
# ==========================================
embedding_fn = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# ==========================================
# 4. READ CSV AND CREATE DOCUMENTS
# ==========================================
print(f"\n📄 Reading {CSV_FILE}...")

documents = []
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        parts = []
        parts.append(f"Name: {row['place_name']}")
        parts.append(f"Address: {row['formatted_address']}")
        if row.get('tags'):
            parts.append(f"Tags: {row['tags']}")
        if row.get('review_1'):
            parts.append(f"Review: {row['review_1']}")
        if row.get('review_2'):
            parts.append(f"Review: {row['review_2']}")
        if row.get('review_3'):
            parts.append(f"Review: {row['review_3']}")

        page_content = "\n".join(parts)

        metadata = {
            "name": row.get('place_name', 'Unknown'),
            "category": row.get('tags', 'Food'),
            "address": row.get('formatted_address', ''),
            "rating": float(row['rating']) if row.get('rating') else 0.0,
            "user_rating_count": int(float(row['user_rating_count'])) if row.get('user_rating_count') else 0,
            "business_status": row.get('business_status', ''),
            "latitude": float(row['latitude']) if row.get('latitude') else 0.0,
            "longitude": float(row['longitude']) if row.get('longitude') else 0.0,
            "phone": row.get('national_phone_number', ''),
            "website": row.get('website_uri', ''),
            "gmaps_uri": row.get('gmaps_uri', ''),
            "gmaps_place_id": row.get('gmaps_place_id', ''),
        }

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

print(f"   ✅ Loaded {len(documents)} restaurant records.")

# ==========================================
# 5. UPLOAD IN BATCHES
# ==========================================
print(f"\n🚀 Uploading to collection '{COLLECTION_NAME}' in batches of {BATCH_SIZE}...")

total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    batch_num = (i // BATCH_SIZE) + 1

    if i == 0:
        vector_db = Chroma.from_documents(
            documents=batch,
            embedding=embedding_fn,
            client=chroma_auth_client,
            collection_name=COLLECTION_NAME
        )
    else:
        vector_db.add_documents(batch)

    print(f"   📦 Batch {batch_num}/{total_batches} uploaded ({len(batch)} docs)")

# ==========================================
# 6. VERIFY
# ==========================================
print(f"\n✅ Done! Collection '{COLLECTION_NAME}' repopulated with {len(documents)} documents.")

collection = chroma_auth_client.get_collection(COLLECTION_NAME)
count = collection.count()
print(f"   📊 Verification: {count} documents in collection.")

final_collections = [c.name for c in chroma_auth_client.list_collections()]
print(f"   📦 All collections: {final_collections}")
