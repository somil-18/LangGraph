from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')


llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash-lite',
    api_key = api_key
)


class CB_State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph = StateGraph(CB_State)


def chat(state: CB_State) -> CB_State:
    # take query
    query = state['messages']

    # response
    response = llm.invoke(query)

    return {'messages': [response]}


graph.add_node('chat', chat)


graph.add_edge(START, 'chat')
graph.add_edge('chat', END)


checkpointer = MemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)


print("********* WELCOME **********")
while True:
    user_input = input('User: ')

    if user_input.strip().lower() in ['exit', 'bye']:
        break

    config = {'configurable': {'thread_id': 'thread-1'}}
    answer = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)

    print(answer)
