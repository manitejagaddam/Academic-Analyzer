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


answer_extractor = AssistantAgent(
    name = "answer_extractor",
    model_client= open_router_agent,
    system_message=prompt
)






prompt2 = """
You are an educational performance analysis agent.

Given student answers with marks, analyze:
1. The best-answered question and why it scored well.
2. The weakest-answered question and why it failed.
3. The student's inferred strengths and weaknesses.
4. Recommendations to improve.

Input format:
[
  {
    "question_no": 1,
    "student_answer": "...",
    "marks_awarded": 4
  },
  ...
]

Output format:
{
  "best_answer": {
    "question_no": 2,
    "reason": "Clear structure, strong argument, well-supported explanation."
  },
  "worst_answer": {
    "question_no": 5,
    "reason": "Vague answer with missing key points and poor grammar."
  },
  "strengths": [
    "Shows good conceptual clarity",
    "Writes structured theoretical responses"
  ],
  "weaknesses": [
    "Poor in applying concepts to practical examples",
    "Weak grammar under pressure"
  ],
  "recommendations": [
    "Practice more real-life application-based questions",
    "Work on time-bound written articulation"
  ]
}
"""

answer_analyser = AssistantAgent(
    name = "answer_Analyser",
    model_client= open_router_agent,
    system_message=prompt2
)









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