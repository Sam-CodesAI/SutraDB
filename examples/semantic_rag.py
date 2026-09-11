"""
End-to-end Retrieval-Augmented Generation (RAG) simulation with SutraDB.
Demonstrates how an AI agent retrieves context from private knowledge bases.
"""

import numpy as np
from sutradb import SutraDB, Document

# 1. Initialize RAG Knowledge Store
db = SutraDB(persist_directory="./rag_data")
kb = db.get_or_create_collection(name="company_handbook", dimension=6, metric="cosine")

# 2. Knowledge chunks with simulated embeddings (6-dimensional)
knowledge_chunks = [
    {
        "id": "kb_eng_01",
        "vector": [0.92, 0.10, 0.05, 0.01, 0.02, 0.00],
        "text": "Production deployments require approvals from two staff engineers and a passing green CI pipeline.",
        "metadata": {"department": "engineering", "access_level": "internal", "version": 2026}
    },
    {
        "id": "kb_sec_01",
        "vector": [0.85, 0.40, 0.02, 0.00, 0.01, 0.05],
        "text": "All API endpoints must enforce mTLS and token revocation checks within 50ms of session invalidation.",
        "metadata": {"department": "security", "access_level": "restricted", "version": 2026}
    },
    {
        "id": "kb_hr_01",
        "vector": [0.05, 0.02, 0.88, 0.15, 0.00, 0.00],
        "text": "Employees receive 25 days of paid annual leave plus floating public holidays in their home timezone.",
        "metadata": {"department": "hr", "access_level": "public", "version": 2025}
    },
    {
        "id": "kb_fin_01",
        "vector": [0.10, 0.00, 0.20, 0.90, 0.05, 0.00],
        "text": "Business travel expenses over $500 require prior written authorization from the regional finance director.",
        "metadata": {"department": "finance", "access_level": "internal", "version": 2026}
    }
]

kb.insert(knowledge_chunks)
print(f"📚 Indexed {kb.count()} handbook chunks into SutraDB")

# 3. Simulate an Agent prompt: "How do we ship code to production safely?"
# Query vector aligned with engineering/security coordinates:
agent_query_vector = [0.90, 0.15, 0.05, 0.00, 0.00, 0.00]
agent_query_text = "deployments production release"

print("\n🔍 Agent Query: 'How do we ship code to production safely?'")
print("🔒 Security Constraint: Only retrieve documents with department in ['engineering', 'security']")

matched_chunks = kb.query(
    vector=agent_query_vector,
    text=agent_query_text,
    filter={"department": {"$in": ["engineering", "security"]}},
    top_k=2,
    hybrid=True
)

print("\n--- Retrieved RAG Context Chunks ---")
for i, chunk in enumerate(matched_chunks, start=1):
    print(f"\n[Context #{i}] (Relevance Score: {chunk.score:.4f})")
    print(f"Source ID: {chunk.id} | Department: {chunk.metadata['department']}")
    print(f"Text: \"{chunk.text}\"")

print("\n🤖 Prompt Augmented with SutraDB Ground Truth -> Ready for Gemini / LLM Generation!")
