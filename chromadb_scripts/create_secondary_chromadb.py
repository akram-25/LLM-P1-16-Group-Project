"""
Populate the 'foodkaki_restaurants_secondary' collection in PostgreSQL (pgvector)
using food_places/food_places_secondary.csv data.

Usage: python chromadb_scripts/create_secondary_chromadb.py
"""

import os
import csv
from pathlib import Path
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ==========================================
# 1. LOAD CONFIG
# ==========================================
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / 'apikeys.env')

PG_CONNECTION = (
    f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

CSV_FILE = ROOT / 'food_places' / 'food_places_secondary.csv'
COLLECTION_NAME = "foodkaki_restaurants_secondary"
BATCH_SIZE = 50

# ==========================================
# 2. SETUP EMBEDDING FUNCTION
# ==========================================
print("🔗 Connecting to PostgreSQL...")

embedding_fn = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# ==========================================
# 3. READ CSV AND CREATE DOCUMENTS
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

        documents.append(Document(page_content=page_content, metadata=metadata))

print(f"   ✅ Loaded {len(documents)} restaurant records.")

# ==========================================
# 4. UPLOAD IN BATCHES
# ==========================================
print(f"\n🚀 Uploading to '{COLLECTION_NAME}' in batches of {BATCH_SIZE}...")

total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
vector_db = None

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    batch_num = (i // BATCH_SIZE) + 1

    if i == 0:
        vector_db = PGVector.from_documents(
            documents=batch,
            embedding=embedding_fn,
            connection=PG_CONNECTION,
            collection_name=COLLECTION_NAME,
            pre_delete_collection=True,
        )
    else:
        vector_db.add_documents(batch)

    print(f"   📦 Batch {batch_num}/{total_batches} uploaded ({len(batch)} docs)")

print(f"\n✅ Done! '{COLLECTION_NAME}' populated with {len(documents)} documents.")
