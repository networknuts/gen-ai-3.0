import asyncio
import json
from openai import OpenAI

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


# ---------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------
client = OpenAI()
SYSTEM_PROMPT = """
You are an AI assistant with access to external tools.

If the user asks about:
- factual information
- wikipedia topics
- internet searches
- document knowledge

You MUST call the appropriate tool.

Do not answer from memory when a tool is available.
"""


# ---------------------------------------------------------
# Convert MCP Tool → OpenAI Tool Schema
# ---------------------------------------------------------
def convert_tool(tool):
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,
    }


# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
async def main():

    query = "check wikipedia on info about donald trump. "

    # -----------------------------------------------------
    # Connect to MCP Server
    # -----------------------------------------------------
    async with streamable_http_client("http://localhost:8000/mcp") as (
        read_stream,
        write_stream,
        _,
    ):

        async with ClientSession(read_stream, write_stream) as session:

            await session.initialize()

            # -------------------------------------------------
            # AUTO DISCOVER TOOLS
            # -------------------------------------------------
            tool_result = await session.list_tools()
            tools = tool_result.tools

            print("\nDiscovered tools:\n")

            for t in tools:
                print("-", t.name)

            openai_tools = [convert_tool(t) for t in tools]

            # -------------------------------------------------
            # Ask LLM
            # -------------------------------------------------
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": query}]
                    }
                ],
                tools=openai_tools
            )

            # -------------------------------------------------
            # Detect Tool Call
            # -------------------------------------------------
            tool_call = None

            for item in response.output:
                if item.type == "function_call":
                    tool_call = item
                    break

            # -------------------------------------------------
            # Execute Tool
            # -------------------------------------------------
            if tool_call:

                tool_name = tool_call.name
                args = json.loads(tool_call.arguments)

                print(f"\nLLM selected tool: {tool_name}\n")

                result = await session.call_tool(tool_name, args)

                print("\nTool Result:\n")

                # MCP returns content blocks
                for c in result.content:
                    if hasattr(c, "text"):
                        print(c.text)
                    else:
                        print(c)

            else:

                print("\nLLM Answer:\n")
                print(response.output_text)


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
