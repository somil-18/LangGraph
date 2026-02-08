from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')


# define state
class LLMState(TypedDict):
    question: str
    answer: str


# define LLm model
llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash-lite',
    api_key = api_key
)


# define function
def llm_qa(state: LLMState) -> LLMState:
    question = state['question']

    prompt = f'Answer this question \n {question}'

    answer = llm.invoke(prompt).content

    state['answer'] = answer

    return state


# define your graph
graph = StateGraph(LLMState)


# define nodes
graph.add_node('llm_qa', llm_qa)


# define edges
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)


# complie graph
workflow = graph.compile()


# execute graph
inital_state = {'question': 'How far is Delhi from Dubai by air?'}
final_state = workflow.invoke(inital_state)
print(final_state) 


# visualize the grpah
print(workflow.get_graph().print_ascii())

