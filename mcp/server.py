# server.py

from mcp.server.fastmcp import FastMCP
import requests
import wikipedia
from qdrant_client import QdrantClient
from openai import OpenAI

# -----------------------------
# MCP SERVER
# -----------------------------

mcp = FastMCP("research_server")

openai_client = OpenAI()


# -----------------------------
# WIKIPEDIA TOOL
# -----------------------------

@mcp.tool()
def wikipedia_search(topic: str) -> str:
    """
    Get wikipedia summary
    """

    try:
        return wikipedia.summary(topic, sentences=10)
    except Exception as e:
        return str(e)


# -----------------------------
# QDRANT VECTOR SEARCH
# -----------------------------

@mcp.tool()
def qdrant_search(
    query: str,
    qdrant_url: str,
    collection_name: str,
    embedding_model: str = "text-embedding-3-small"
) -> str:

    emb = openai_client.embeddings.create(
        model=embedding_model,
        input=query
    )

    vector = emb.data[0].embedding

    qdrant = QdrantClient(url=qdrant_url)

    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=3
    )

    results = []

    for h in hits:
        if h.payload and "text" in h.payload:
            results.append(h.payload["text"])

    return "\n\n".join(results)


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
