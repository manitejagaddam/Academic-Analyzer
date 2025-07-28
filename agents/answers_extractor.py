from autogen_agentchat.agents import AssistantAgent
# from open_router_agent import open_router_agent
import requests 
from autogen_core import Image as AGImage
from PIL import Image
from io import BytesIO
from autogen_agentchat.messages import MultiModalMessage


from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")


open_router_agent = OpenAIChatCompletionClient(
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


# prompt_path = "../prompts/answer_extractor.txt" 

# with open(prompt_path, "r", encoding="utf-8") as f:
#     prompt = f.read()


prompt = """
You are an intelligent answer extractor agent.

Given an Image of a handwritten exam sheet, identify:
1. Each question and its answer.
2. Marks awarded (if visible).
3. Whether the answer is complete, partial, or blank.

Output the result in this format:

[
  {
    "question_no": 1,
    "question_text": "Explain Newton’s second law.",
    "student_answer": "Force equals mass times acceleration...",
    "marks_awarded": 4,
    "status": "complete"
  },
  ...
]
"""

def answer_extractor():
    # response = requests.get('https://picsum.photos/id/15/200/300') # 23 for the image of folks
    # pil_image = Image.open(BytesIO(response.content))
    # ag_image = AGImage(pil_image)

    # multi_modal_msg = MultiModalMessage(
    #     content = [prompt,ag_image],    
    #     source='user'
    # )
    
    agent = AssistantAgent(
        name = "answer_extractor",
        model_client= open_router_agent(),
        system_message=prompt
    )
    
    return agent


