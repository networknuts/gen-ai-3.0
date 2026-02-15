from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI 

# ENVIRONMENT SETUP
load_dotenv()
client = OpenAI()


# EMBEDDING MODEL - SAME AS OF THE VECTOR DB

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large")

# CONNECT TO YOUR VECTOR DB

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="ansible_vectors",
    url="http://localhost:6333",
)

# ACCEPT THE USER INPUT

user_query = input("> ")

# SEARCH THE VECTOR DB FOR USER INPUT SIMILARITY SEARCH

search_results = vector_db.similarity_search(query=user_query)

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

# GENERATE THE ANSWER USING THE SYSTEM PROMPT AS CONTEXT

response = client.responses.create(
    model="gpt-5-nano",
    instructions=SYSTEM_PROMPT, 
    input=user_query
)

print(response.output_text)
