import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient


# -------------------------------------------------------
# ENV + MODEL
# -------------------------------------------------------

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

MONGO_DB = os.getenv("MONGODB_URI")
CLIENT = MongoClient(MONGO_DB)
MAX_RETRIES = 3

# -------------------------------------------------------
# LLM JSON HELPER
# -------------------------------------------------------

def llm_json(prompt: str):
    response = llm.invoke(
        "Return ONLY valid JSON. No markdown.\n\n" + prompt
    ).content.strip()
    return json.loads(response)

# -------------------------------------------------------
# STATE
# -------------------------------------------------------

class CodeState(TypedDict):
    user_request: str
    code: str
    rating: int
    feedback: str
    retries: int
    status: str

# -------------------------------------------------------
# NODE 1: DEVELOPER AGENT
# -------------------------------------------------------

def developer_agent(state: CodeState):

    prompt = f"""
You are a junior Node.js developer.

Write extremely poorly written bad, dirty Node.js software with intentionally wrong practices for:

{state["user_request"]}

If feedback is provided, improve the previous version accordingly.

Previous Code:
{state["code"]}

Feedback:
{state["feedback"]}

Return ONLY the full Node.js code.
"""

    result = llm.invoke(prompt).content

    return {
        "code": result,
        "feedback": ""
    }

# -------------------------------------------------------
# NODE 2: QA AGENT
# -------------------------------------------------------

def qa_agent(state: CodeState):

    prompt = f"""
You are a strict QA engineer.

Evaluate the following Node.js code.

Return JSON only:
{{
  "rating": number between 1-10,
  "feedback": "clear explanation of improvements"
}}

Code:
{state["code"]}
"""

    result = llm_json(prompt)

    return {
        "rating": int(result["rating"]),
        "feedback": result["feedback"]
    }

# -------------------------------------------------------
# STATE UPDATES
# -------------------------------------------------------

def set_approved(state: CodeState):
    return {"status": "approved"}

def set_failed(state: CodeState):
    return {"status": "failed"}

def increment_retry(state: CodeState):
    return {"retries": state["retries"] + 1}

# -------------------------------------------------------
# ROUTER
# -------------------------------------------------------

def check_rating(state: CodeState):

    if state["rating"] >= 7:
        return "approved"

    if state["retries"] >= MAX_RETRIES:
        return "failed"

    return "retry"

# -------------------------------------------------------
# BUILD GRAPH
# -------------------------------------------------------

graph = StateGraph(CodeState)

graph.add_node("developer", developer_agent)
graph.add_node("qa", qa_agent)
graph.add_node("retry_increment", increment_retry)
graph.add_node("approved_node", set_approved)
graph.add_node("failed_node", set_failed)

graph.set_entry_point("developer")

graph.add_edge("developer", "qa")

graph.add_conditional_edges(
    "qa",
    check_rating,
    {
        "approved": "approved_node",
        "retry": "retry_increment",
        "failed": "failed_node"
    }
)

graph.add_edge("retry_increment", "developer")
graph.add_edge("approved_node", END)
graph.add_edge("failed_node", END)

# -------------------------------------------------------
# ADD MONGODB CHECKPOINTING
# -------------------------------------------------------

checkpointer = MongoDBSaver(CLIENT)

app = graph.compile(checkpointer=checkpointer)

# -------------------------------------------------------
# RUN
# -------------------------------------------------------

if __name__ == "__main__":

    thread_id = "nodejs_agent_1"

    # Check if thread already exists in Mongo
    existing = checkpointer.get({"configurable": {"thread_id": thread_id}})

    try:
        if existing:
            print("Resuming from checkpoint...\n")

            result = app.invoke(
                {},
                config={"configurable": {"thread_id": thread_id}}
            )

        else:
            user_input = input("What Node.js software should be built? ")

            result = app.invoke(
                {
                    "user_request": user_input,
                    "code": "",
                    "rating": 0,
                    "feedback": "",
                    "retries": 0,
                    "status": "running"
                },
                config={"configurable": {"thread_id": thread_id}}
            )

        print("\n--- FINAL RESULT ---\n")
        print("Status:", result["status"])
        print("Final Rating:", result["rating"])
        print("Retries Used:", result["retries"])
        print("\nFinal Code:\n")
        print(result["code"])

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! State saved to MongoDB.")
