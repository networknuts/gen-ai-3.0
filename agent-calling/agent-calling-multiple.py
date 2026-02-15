import requests
from openai import OpenAI 
from dotenv import load_dotenv
import os
import json
import subprocess

# SETUP MY ENVIRONMENT
load_dotenv()
client = OpenAI()


# TOOL INDEX

def get_weather(zip_code: str):
    apikey = os.getenv("WEATHER_API_KEY")
    country = "in"
    url = f"https://api.openweathermap.org/data/2.5/weather?zip={zip_code},{country}&appid={apikey}"
    result = requests.get(url)
    response = result.json()
    return response

def run_shell(command: str):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout

# TOOL SCHEMA

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather data of a city by providing the zip code",
        "parameters": {
            "type": "object",
            "properties": {"zip_code": {"type": "string","description": "the zip code of the city"}},
            "required": ["zip_code"]
        }
    },
    {
        "type": "function",
        "name": "run_shell",
        "description": "run shell commands",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
    }
]

# FIRST LLM CALL

user_input = """
provide me important info regarding my system
"""

response = client.responses.create(
    model="gpt-4.1",
    input=user_input,
    tools=tools
)

# EXECUTING OF TOOL AND COLLECTION OF TOOL RESPONSE

tool_outputs = []

for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)

        if item.name == "get_weather":
            result = get_weather(args["zip_code"])
        elif item.name == "run_shell":
            result = run_shell(args["command"])
        else:
            result = "unknown tool"

        tool_outputs.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps({"result": result})
        })


# SECOND LLM CALL

final_response = client.responses.create(
    model = "gpt-4.1",
    input = tool_outputs,
    previous_response_id = response.id
)

print(final_response.output_text)
