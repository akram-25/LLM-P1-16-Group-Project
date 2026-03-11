"""
Inspect both ChromaDB collections - view stats and sample data.
Usage: python inspect_chromadb.py
"""

import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

load_dotenv('apikeys.env')

print("=" * 60)
print("  ChromaDB Collection Inspector")
print("=" * 60)

client = chromadb.HttpClient(
    host=os.getenv('CHROMA_SERVER_URL'),
    port=443,
    ssl=True,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials=os.getenv('CHROMA_API_TOKEN'),
        anonymized_telemetry=False
    )
)

client.heartbeat()
print("\n✅ Connected to ChromaDB server\n")

collections = client.list_collections()
print(f"📦 Total collections: {len(collections)}")
print("-" * 60)

for col_info in collections:
    col = client.get_collection(col_info.name)
    count = col.count()
    print(f"\n📁 Collection: {col_info.name}")
    print(f"   Documents: {count}")

    # Peek at first 3 documents
    if count > 0:
        sample = col.peek(limit=3)
        print(f"   Sample entries:")
        for i, doc_id in enumerate(sample['ids']):
            meta = sample['metadatas'][i] if sample['metadatas'] else {}
            content = (sample['documents'][i][:120] + "...") if sample['documents'] and sample['documents'][i] else "N/A"
            name = meta.get('name', 'Unknown')
            rating = meta.get('rating', 'N/A')
            address = meta.get('address', 'N/A')
            print(f"\n   [{i+1}] {name}")
            print(f"       Rating: {rating} | Address: {address}")
            print(f"       Preview: {content}")

    print("-" * 60)
