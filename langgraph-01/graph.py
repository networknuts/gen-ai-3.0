import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# -------------------------------------------------------
# ENV + MODEL
# -------------------------------------------------------

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",  # lightweight + fast
    temperature=0
)

# -------------------------------------------------------
# STATE DEFINITION
# -------------------------------------------------------

class SupportState(TypedDict):
    user_query: str
    intent: str
    response: str

# -------------------------------------------------------
# NODE 1: INTENT CLASSIFIER
# -------------------------------------------------------

def classify_intent(state: SupportState):
    prompt = f"""
    Classify the user query into one of these categories:
    - password_reset
    - order_tracking
    - refund

    Only return the category name.

    User query: {state["user_query"]}
    """

    result = llm.invoke(prompt)

    return {"intent": result.content.strip().lower()}

# -------------------------------------------------------
# NODE 2A: PASSWORD RESET HANDLER
# -------------------------------------------------------

def handle_password(state: SupportState):
    return {
        "response": (
            "To reset your password, click on 'Forgot Password' "
            "on the login page and follow the instructions sent to your email."
        )
    }

# -------------------------------------------------------
# NODE 2B: ORDER TRACKING HANDLER
# -------------------------------------------------------

def handle_order(state: SupportState):
    return {
        "response": (
            "You can track your order from the 'My Orders' section "
            "in your account dashboard."
        )
    }

# -------------------------------------------------------
# NODE 2C: REFUND HANDLER
# -------------------------------------------------------

def handle_refund(state: SupportState):
    return {
        "response": (
            "Refunds can be requested within 7 days of delivery. "
            "Please visit the 'Returns & Refunds' section in your account."
        )
    }

# -------------------------------------------------------
# ROUTING FUNCTION
# -------------------------------------------------------

def route_intent(state: SupportState):
    if state["intent"] == "password_reset":
        return "password_node"
    elif state["intent"] == "order_tracking":
        return "order_node"
    elif state["intent"] == "refund":
        return "refund_node"
    else:
        return END

# -------------------------------------------------------
# BUILD GRAPH
# -------------------------------------------------------

graph = StateGraph(SupportState)

# Add nodes
graph.add_node("classifier", classify_intent)
graph.add_node("password_node", handle_password)
graph.add_node("order_node", handle_order)
graph.add_node("refund_node", handle_refund)

# Entry point
graph.set_entry_point("classifier")

# Conditional routing
graph.add_conditional_edges(
    "classifier",
    route_intent
)

# End edges
graph.add_edge("password_node", END)
graph.add_edge("order_node", END)
graph.add_edge("refund_node", END)

# Compile graph
app = graph.compile()

# -------------------------------------------------------
# RUN APP
# -------------------------------------------------------

if __name__ == "__main__":
    user_input = input("User: ")

    result = app.invoke({
        "user_query": user_input,
        "intent": "",
        "response": ""
    })

    print("\nBot:", result["response"])
