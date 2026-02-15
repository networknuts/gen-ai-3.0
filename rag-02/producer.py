import redis
import uuid


# REDIS CONNECTION

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

# PUSH QUERY TO THE QUEUE

def enqueue_query(query: str):
    job_id = str(uuid.uuid4())

    payload = {
        "job_id": job_id,
        "query": query
    }
    redis_client.rpush("rag:requests", str(payload))
    return job_id

user_query = input("> ")
job = enqueue_query(user_query)

print("query sent to queue")
print(job)

