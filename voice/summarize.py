from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def load_system_prompt(path):
    with open(path, "r") as f:
        return f.read().strip()
    
def summarize(notes, system_prompt):
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": notes}
        ]
    )
    return response.output_text

f = open("meeting-raw-notes.txt","r")
notes = f.read()
f.close()

system_prompt = load_system_prompt("system_prompt.txt")

summary = summarize(notes,system_prompt)
print(summary)               
