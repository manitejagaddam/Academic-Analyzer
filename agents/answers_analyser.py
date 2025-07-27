from autogen_agentchat.agents import AssistantAgent
# from open_router_agent import open_router_agent
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

# prompt_path = "../prompts/answers_analysis.txt" 

# with open(prompt_path, "r", encoding="utf-8") as f:
#     prompt = f.read()

prompt = """
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

def answer_analyser():
    agent = AssistantAgent(
        name = "answer Analyser",
        model_client= open_router_agent,
        system_message=prompt
    )
    
    return agent