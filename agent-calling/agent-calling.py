import json
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------
# ENV + CLIENT
# ---------------------------------------------------------

load_dotenv()
client = OpenAI()

# ---------------------------------------------------------
# TOOLS (PYTHON)
# ---------------------------------------------------------

def get_weather(city: str) -> str:
    fake_weather = {
        "delhi": "🌤️ 32°C, clear",
        "london": "🌧️ 12°C, rainy",
        "new york": "☁️ 18°C, cloudy",
    }
    return fake_weather.get(city.lower(), "Weather not found")


def run_shell(command: str) -> str:
    if not command.startswith(("echo", "mkdir")):
        return "❌ Command not allowed"

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "✅ Command executed"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# TOOL SCHEMAS
# ---------------------------------------------------------

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "name": "run_shell",
        "description": "Run safe shell commands like echo or mkdir",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

# ---------------------------------------------------------
# 1️⃣ FIRST MODEL CALL
# ---------------------------------------------------------

response = client.responses.create(
    model="gpt-4.1",
    input="what is the weather in delhi?",
    tools=tools,
)

# ---------------------------------------------------------
# 2️⃣ EXECUTE TOOL + COLLECT TOOL OUTPUT
# ---------------------------------------------------------

tool_outputs = []

for item in response.output:
    if item.type == "function_call":

        args = json.loads(item.arguments)

        if item.name == "get_weather":
            result = get_weather(args["city"])
        elif item.name == "run_shell":
            result = run_shell(args["command"])
        else:
            result = "Unknown tool"

        tool_outputs.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps({"result": result}),
        })

# ---------------------------------------------------------
# 3️⃣ SECOND MODEL CALL (CONTINUATION)
# ---------------------------------------------------------

final_response = client.responses.create(
    model="gpt-4.1",
    previous_response_id=response.id,  # 🔑 THIS IS THE FIX
    input=tool_outputs,
)

print(final_response.output_text)
