import hashlib
import uuid

import redis
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# -----------------------------
# Setup
# -----------------------------
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
qdrant = QdrantClient(host="localhost", port=6333)
client = OpenAI()

COLLECTION = "cache"


# -----------------------------
# Hashing
# -----------------------------
def make_key(prompt: str) -> str:
    normalized = prompt.strip().lower()
    hashed = hashlib.sha256(normalized.encode()).hexdigest()
    return f"cache:{hashed}"


# -----------------------------
# Embedding
# -----------------------------
def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding


# -----------------------------
# Init Qdrant
# -----------------------------
def init_collection(vector_size):
    try:
        qdrant.get_collection(COLLECTION)
    except:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


# -----------------------------
# LLM
# -----------------------------
def ask_llm(prompt):
    res = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )
    return res.output_text


# -----------------------------
# Semantic search
# -----------------------------
def search_cache(embedding):
    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=embedding,
        limit=1
    )

    if len(res.points) == 0:
        return None

    point = res.points[0]

    if point.score > 0.9:
        return point.payload["answer"]

    return None


# -----------------------------
# Save to Qdrant
# -----------------------------
def save_cache(prompt, embedding, answer):
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "prompt": prompt,
                    "answer": answer
                }
            )
        ]
    )


# -----------------------------
# Main logic
# -----------------------------
def get_answer(prompt):
    key = make_key(prompt)

    # 1. Redis exact match
    cached = r.get(key)
    if cached:
        print("REDIS HIT")
        return cached

    # 2. Embedding
    emb = get_embedding(prompt)

    # 3. Init collection
    init_collection(len(emb))

    # 4. Semantic cache
    semantic = search_cache(emb)
    if semantic:
        print("QDRANT HIT")
        r.set(key, semantic)
        return semantic

    # 5. LLM fallback
    print("LLM CALL")
    answer = ask_llm(prompt)

    # 6. Save
    r.set(key, answer)
    save_cache(prompt, emb, answer)

    return answer


# -----------------------------
# Run loop
# -----------------------------
while True:
    q = input("\nYou: ")

    if q == "exit":
        break

    print("AI:", get_answer(q))
