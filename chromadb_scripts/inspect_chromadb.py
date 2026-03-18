"""
Inspect both pgvector collections - view stats and sample data.
Usage: python chromadb_scripts/inspect_chromadb.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / 'apikeys.env')

PG_CONNECTION = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

COLLECTIONS = ["foodkaki_restaurants", "foodkaki_restaurants_secondary"]

print("=" * 60)
print("  pgvector Collection Inspector")
print("=" * 60)

with psycopg.connect(PG_CONNECTION) as conn:
    print("\n✅ Connected to PostgreSQL\n")

    for collection_name in COLLECTIONS:
        print(f"\n📁 Collection: {collection_name}")

        # Count documents in this collection
        row = conn.execute("""
            SELECT COUNT(*) FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
        """, (collection_name,)).fetchone()

        count = row[0] if row else 0
        print(f"   Documents: {count}")

        if count > 0:
            samples = conn.execute("""
                SELECT e.cmetadata, e.document FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                LIMIT 3
            """, (collection_name,)).fetchall()

            print(f"   Sample entries:")
            for i, (meta, content) in enumerate(samples):
                name = meta.get('name', 'Unknown') if meta else 'Unknown'
                rating = meta.get('rating', 'N/A') if meta else 'N/A'
                address = meta.get('address', 'N/A') if meta else 'N/A'
                preview = (content[:120] + "...") if content and len(content) > 120 else content
                print(f"\n   [{i+1}] {name}")
                print(f"       Rating: {rating} | Address: {address}")
                print(f"       Preview: {preview}")

        print("-" * 60)
