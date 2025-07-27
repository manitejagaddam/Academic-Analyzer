import asyncio
from PIL import Image
from io import BytesIO
import requests
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import Image as AGImage   

# agents
from agents.answers_analyser import answer_analyser
from agents.answers_extractor import answer_extractor


team = RoundRobinGroupChat(
    participants=[answer_extractor, answer_analyser],
    max_turns=2
)

image_path = "https://images.app.goo.gl/C3Ppk"

async def run_team():
    response = requests.get(image_path) # 23 for the image of folks
    pil_image = Image.open(BytesIO(response.content))
    ag_image = AGImage(pil_image)

    
    task = TextMessage(content=ag_image, source='user')
    result = await team.run(task=task)
    return result.messages