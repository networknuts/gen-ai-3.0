import redis 
import time

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

job_id = input("Enter Job ID: ")

while True:
    result = redis_client.get(f"rag:response:{job_id}")
    if result:
        print(f"Output:\n ")
        print(result)
        break
    print("Waiting to get result")
    time.sleep(2)
