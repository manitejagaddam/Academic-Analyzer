from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import os 
from dotenv import load_dotenv


load_dotenv()

open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")


def open_router_agent():
    agent = OpenAIChatCompletionClient(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-r1-0528:free",
        api_key=open_router_api_key,
        model_info={
            "family": "deepseek",
            "vision": True,
            "function_calling": True,
            "json_output": False
        }
    )
    return agent