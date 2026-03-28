import hashlib
import redis
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
client = OpenAI()


# -----------------------------
# Hashing
# -----------------------------
def make_key(prompt: str) -> str:
    normalized = prompt.strip().lower()
    hashed = hashlib.sha256(normalized.encode()).hexdigest()
    return f"cache:{hashed}"


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
# Main logic
# -----------------------------
def get_answer(prompt):
    key = make_key(prompt)

    # Check Redis
    cached = r.get(key)
    if cached:
        print("CACHE HIT (Redis)")
        return cached

    # Call LLM
    print("LLM CALL")
    answer = ask_llm(prompt)

    # Save
    r.set(key, answer)

    return answer


# -----------------------------
# Run loop
# -----------------------------
while True:
    q = input("\nYou: ")

    if q == "exit":
        break

    print("AI:", get_answer(q))
