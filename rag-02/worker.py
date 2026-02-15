from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI
import redis
import ast


# ENVIRONMENT SETUP
load_dotenv()
client = OpenAI()


# REDIS SETUP
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

# EMBEDDING MODEL - SAME AS OF THE VECTOR DB

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large")

# CONNECT TO YOUR VECTOR DB

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="ansible_vectors",
    url="http://localhost:6333",
)

print("Worker started. Waiting for jobs to process.")

while True:
    queue_name, raw_payload = redis_client.blpop("rag:requests")

    payload = ast.literal_eval(raw_payload)
    job_id = payload["job_id"]
    query = payload["query"]

    print(f"Processing Job: {job_id}")

    # semantic search in the vector db
    search_results = vector_db.similarity_search(query=query)
    context = []

    for item in search_results:
        block = f"""
    Page Content: {item.page_content}
    Page Number: {item.metadata.get("page","NA")}
    """
        context.append(block.strip())

    final_context = "----\n\n".join(context)

    SYSTEM_PROMPT = f"""
You are a RAG AI Assistant Chatbot.
You have been given extracted content from a PDF document:
The extracted content contains the following:
- The text content
- The PDF page number

Answer the question of the user using ONLY this provided information.

If the answer exists:
- Respond in a clear and concise manner
- Mention the relevant PDF page number so the user can read more

If the answer does not exist:
- Clearly say that the information is not available in your knowledge base

DO NOT ADD OUTSIDE KNOWLEDGE IN ANY SCENARIO.

Context:
{final_context}
"""

    response = client.responses.create(
        model="gpt-5-nano",
        instructions=SYSTEM_PROMPT, 
        input=query
    )

    answer = response.output_text

    redis_client.set(
        f"rag:response:{job_id}",
        answer,
        ex=3600
    )

    print(f"Job {job_id} completed.")
