from openai import OpenAI
client = OpenAI()

previous_story = "A gentle unicorn drifted across a moonlit meadow, lulling the stars to sleep."

response = client.responses.create(
    model="gpt-5-nano",
    input=[
        {
            "role": "system",
            "content": "You are a creative storyteller who writes calm, soothing bedtime stories."
        },
        {
            "role": "assistant",
            "content": previous_story
        },
        {
            "role": "user",
            "content": "Rewrite the story but make it slightly more magical."
        }
    ]
)

print(response.output_text)
