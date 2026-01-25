from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input=[
        {
            "role": "system",
            "content": "You are a creative storyteller who writes calm, soothing bedtime stories."
        },
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn."
        }
    ]
)

print(response.output_text)
