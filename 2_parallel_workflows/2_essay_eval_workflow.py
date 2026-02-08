from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import operator

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

# structured output
class Llm_struct_op(BaseModel):
    feedback: str = Field(description="Give honest feedback")
    score: int = Field(description="Score out of 10", ge=0, le=10)

struct_llm = llm.with_structured_output(Llm_struct_op)

# state
class Essay_state(TypedDict):
    essay: str
    cot_feedback: str
    doa_feedback: str
    g_feedback: str
    summarized_feedback: str
    scores: Annotated[list[int], operator.add]
    avg_score: float

graph = StateGraph(Essay_state)

# nodes
def cot(state: Essay_state):
    result = struct_llm.invoke(
        f"Evaluate clarity of thought:\n{state['essay']}"
    )
    return {
        "cot_feedback": result.feedback,
        "scores": [result.score]
    }

def doa(state: Essay_state):
    result = struct_llm.invoke(
        f"Evaluate depth of analysis:\n{state['essay']}"
    )
    return {
        "doa_feedback": result.feedback,
        "scores": [result.score]
    }

def g(state: Essay_state):
    result = struct_llm.invoke(
        f"Evaluate grammar:\n{state['essay']}"
    )
    return {
        "g_feedback": result.feedback,
        "scores": [result.score]
    }

#  MERGE BARRIER
def merge(state: Essay_state):
    return {}

def final_eval(state: Essay_state):
    avg_score = sum(state["scores"]) / len(state["scores"])

    summary_prompt = f"""
Summarize the following feedback:

Clarity: {state['cot_feedback']}
Depth: {state['doa_feedback']}
Grammar: {state['g_feedback']}
"""

    result = llm.invoke(summary_prompt)

    return {
        "summarized_feedback": result.content,
        "avg_score": round(avg_score, 2)
    }

# graph wiring
graph.add_node("cot", cot)
graph.add_node("doa", doa)
graph.add_node("g", g)
graph.add_node("merge", merge)
graph.add_node("final_eval", final_eval)

graph.add_edge(START, "cot")
graph.add_edge(START, "doa")
graph.add_edge(START, "g")

graph.add_edge("cot", "merge")
graph.add_edge("doa", "merge")
graph.add_edge("g", "merge")

graph.add_edge("merge", "final_eval")
graph.add_edge("final_eval", END)

workflow = graph.compile()

# proper initial state
essay = '''
Diwali, also known as Deepavali, is one of the grandest festivals of India. It is celebrated with enthusiasm in every state, cutting across religions and communities. The word “Deepavali” means a row of lamps, and the festival truly lives up to its name as millions of diyas light up homes, streets, and temples.

The festival is celebrated for five days, beginning with Dhanteras and ending with Bhai Dooj. The main day of Diwali is marked by the worship of Goddess Lakshmi and Lord Ganesha. Families clean and decorate their homes, wear new clothes, and prepare delicious dishes and sweets. The night sky is lit with fireworks, while the streets glow with festive lights.

According to mythology, Diwali celebrates the return of Lord Rama to Ayodhya after defeating Ravana. The people of Ayodhya lit diyas to welcome him, symbolizing the triumph of good over evil. Some communities also connect Diwali with Goddess Lakshmi and the harvest season.

This Diwali festival essay shows us that the festival is not only about lights and crackers but also about spreading kindness, happiness, and positivity. It is a time to strengthen family bonds, share joy with neighbors, and remember that good always prevails.
'''

initial_state = {
    "essay": essay,
    "cot_feedback": "",
    "doa_feedback": "",
    "g_feedback": "",
    "summarized_feedback": "",
    "scores": [],
    "avg_score": 0.0,
}

final_state = workflow.invoke(initial_state)

print(final_state)
