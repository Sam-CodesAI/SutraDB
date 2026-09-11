"""
Quickstart guide for SutraDB: Initialize, Insert, and Hybrid Query.
"""

from sutradb import SutraDB, Document

# 1. Initialize SutraDB with local disk persistence
db = SutraDB(persist_directory="./sutra_data")

# 2. Create a collection (dimension: 4, metric: cosine)
collection = db.get_or_create_collection(name="products", dimension=4, metric="cosine")

# 3. Insert documents with vector coordinates, lexical text, and structured metadata
collection.insert([
    Document(
        id="prod_01",
        vector=[0.95, 0.05, 0.10, 0.00],
        text="Lightweight running shoes for marathon training",
        metadata={"category": "footwear", "brand": "Nike", "price": 120}
    ),
    Document(
        id="prod_02",
        vector=[0.02, 0.98, 0.05, 0.01],
        text="Waterproof hiking boots for mountain trails",
        metadata={"category": "footwear", "brand": "Columbia", "price": 160}
    ),
    Document(
        id="prod_03",
        vector=[0.88, 0.12, 0.20, 0.00],
        text="Breathable road athletic sneakers",
        metadata={"category": "footwear", "brand": "Adidas", "price": 95}
    )
])

print(f"Total documents indexed: {collection.count()}")

# 4. Perform Hybrid Search with Compound Metadata Filter
# "Find footwear similar to running shoes where price is under $100"
results = collection.query(
    vector=[1.0, 0.0, 0.0, 0.0],
    text="running sneakers",
    filter={"price": {"$lt": 100}},
    top_k=5,
    hybrid=True
)

print("\n--- Search Results ---")
for r in results:
    print(f"ID: {r.id} | Score: {r.score:.4f} | Dense: {r.dense_score:.4f} | BM25: {r.bm25_score:.4f}")
    print(f"Text: {r.text}")
    print(f"Metadata: {r.metadata}\n")
