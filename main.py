import asyncio
from PIL import Image
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import Image as AGImage   

# agents
from agents.answers_analyser import answer_analyser
from agents.answers_extractor import answer_extractor



team = RoundRobinGroupChat(
    participants=[answer_extractor(), answer_analyser()],
    max_turns=2
)

image_path = "Images/test1.png"

async def run_team():
    pil_image = Image.open(image_path)
    ag_image = AGImage(pil_image)

    
    task = MultiModalMessage(content=["give the response accordingly",ag_image], source='user')
    result = await team.run(task=task)
    print(result)
    return result.messages



if __name__ == "__main__":
    asyncio.run(run_team())