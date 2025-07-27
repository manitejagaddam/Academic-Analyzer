import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv
import os


load_dotenv()

open_router_api = os.getenv("OPEN_ROUTER_API_KEY")


# Open Router Assistve Agent
open_router_model_client = OpenAIChatCompletionClient(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528:free",
    api_key=open_router_api,
    model_info={
        "family": "deepseek",
        "vision": True,
        "function_calling": True,
        "json_output": False
    }
)



# Agents :
# 1. analyse the images
# 2. fetch the highest part of the exam sheet paper and the lest scored part of the exam sheet 
# 3. give me the best skills that you had like good in criticalk thinking and give the backing things like lagging in the implementation part


