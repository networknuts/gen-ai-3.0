import os
import json
from typing import TypedDict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ----------------------------------
# ENV
# ----------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------
# NEO4J DRIVER
# ----------------------------------

neo4j_driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "your-password")
)

# ----------------------------------
# STATE
# ----------------------------------

class ChatState(TypedDict):
    user_id: str
    user_query: str
    ai_reply: str
    store_memory: Optional[bool]
    extracted_facts: Optional[list]

# ----------------------------------
# NODE 1: CHAT LLM
# ----------------------------------

def chat_node(state: ChatState):

    response = llm.invoke(state["user_query"])
    state["ai_reply"] = response.content

    print(f"\n🤖 {state['ai_reply']}")
    return state

# ----------------------------------
# NODE 2: MEMORY CLASSIFIER
# ----------------------------------

def memory_classifier_node(state: ChatState):

    prompt = f"""
You are a profile memory classifier.

Determine whether this message contains
long-term personal information about the user.

Return STRICT JSON:

{{
  "store": true or false,
  "facts": [list of extracted durable facts]
}}

Message:
{state["user_query"]}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    decision = json.loads(response.choices[0].message.content)

    state["store_memory"] = decision["store"]
    state["extracted_facts"] = decision.get("facts", [])

    return state

# ----------------------------------
# NODE 3: NEO4J SAVE
# ----------------------------------

def neo4j_save_node(state: ChatState):

    if not state["extracted_facts"]:
        return state

    with neo4j_driver.session() as session:

        for fact in state["extracted_facts"]:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                MERGE (m:Memory {text: $fact})
                MERGE (u)-[:HAS_MEMORY]->(m)
                """,
                user_id=state["user_id"],
                fact=fact
            )

    print("💾 Saved to Neo4j")

    return state

# ----------------------------------
# CONDITIONAL ROUTER
# ----------------------------------

def router(state: ChatState):
    if state["store_memory"]:
        return "neo4j_save"
    return END

# ----------------------------------
# BUILD GRAPH
# ----------------------------------

graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("memory_classifier", memory_classifier_node)
graph.add_node("neo4j_save", neo4j_save_node)

graph.set_entry_point("chat")

graph.add_edge("chat", "memory_classifier")

graph.add_conditional_edges(
    "memory_classifier",
    router,
    {
        "neo4j_save": "neo4j_save",
        END: END
    }
)

graph.add_edge("neo4j_save", END)

app = graph.compile()

# ----------------------------------
# CHAT LOOP
# ----------------------------------

def run_chat():
    user_id = input("User ID: ")

    while True:
        user_query = input("\n> ")
        if user_query.lower() == "exit":
            break

        app.invoke({
            "user_id": user_id,
            "user_query": user_query
        })

if __name__ == "__main__":
    run_chat()
