'''
1. Persistence in LangGraph refers to the ability to save and restore the state of a workflow over time, allowing the workflow to resume execution without losing progress.

2. Problems before persistence:
    - Progress was lost → Every time the graph was run, it restarted from the beginning
    - No memory → Graph could not remember previous conversations, results, or state
    - No resume capability → If the program crashed, the workflow had to restart completely
    - LangGraph workflows were stateless

3. Persistence does not only save the final state, it also saves intermediate states.
    - Example: START → NODE_1 → NODE_2 → END
    - Initially name = 'a'
    - After NODE_1 → name = 'b'
    - After NODE_2 → name = 'c'
    - Persistence saves the state after each node execution, enabling recovery, resume capability, and state history tracking

4. Where are these states saved?
    → In memory (RAM) or in a database/file, depending on the checkpointer used

5. Checkpointer:
    - A checkpointer is the component responsible for saving and loading the graph state
    - It automatically saves the state after each node execution
    - It is the mechanism that enables persistence

6. Types of Checkpointers:
    - MemorySaver → Temporary (stored in RAM, lost when program stops)
    - SQLiteSaver → Persistent (stored in SQLite file)
    - PostgresSaver → Persistent (stored in PostgreSQL database, used in production)

7. Persistence vs Checkpointer:
    - Persistence is the concept (saving and restoring workflow state)
    - Checkpointer is the implementation (tool that performs the saving and loading)

8. thread_id:
    - thread_id is a unique identifier for a workflow session
    - It is used by the checkpointer to save and retrieve the correct workflow state
    - It allows multiple independent sessions to exist simultaneously

9. Why thread_id is needed:
    - Example: Two users using a chatbot → User_A and User_B
    - Both have different conversations
    - thread_id keeps their states separate
    - LangGraph loads the correct state based on thread_id

10. Benefits of Persistence:
    - Short-Term Memory → Workflow remembers previous state
    - Fault Tolerance → Workflow can recover after crashes
    - Human-in-the-Loop (HITL) → Workflow can pause and resume after human input
    - Time Travel → Ability to view and replay past states

11. Resume behavior:
    - LangGraph resumes execution only if the workflow has not reached the END node
    - If the workflow has already reached END, invoking it again with the same thread_id returns the saved state without re-executing the nodes
    - To start a new execution, a new thread_id must be used
'''


from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

from langgraph.checkpoint.memory import MemorySaver


load_dotenv()


model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash-lite',
    api_key = os.getenv('GEMINI_API_KEY')
)


# define state
class JokeState(TypedDict):
    topic: str

    joke: str
    explanation: str


# define graph
graph = StateGraph(JokeState)


# define functions
def joke(state: JokeState) -> JokeState:
    prompt = f'Generate a joke about this topic \n {state["topic"]} \n Rules: \n 1. Use simple and easy to understand english \n 2. Joke should be as double meaning as possible'

    response = model.invoke(prompt).content

    return {'joke': response}

def joke_explanation(state: JokeState) -> JokeState:
    prompt = f'Generate a explanation about this joke \n {state["joke"]}'

    response = model.invoke(prompt).content

    return {'explanation': response}


# define nodes
graph.add_node('joke', joke)
graph.add_node('joke_explanation', joke_explanation)


# define edges
graph.add_edge(START, 'joke')
graph.add_edge('joke', 'joke_explanation')
graph.add_edge('joke_explanation', END)


# make an object of MemorySaver class 
checkpointer = MemorySaver()


# complie the graph with checkpointer
workflow = graph.compile(checkpointer=checkpointer)


# ḍefine threads
config1 = {'configurable': {'thread_id': "1"}}
config2 = {'configurable': {'thread_id': "2"}}


result1 = workflow.invoke({'topic': 'India civic sense'}, config=config1)
print(result1)

result2 = workflow.invoke({'topic': 'India in population'}, config=config2)
print(result2)


# get state
print(workflow.get_state(config1))
print(workflow.get_state(config2))


# get_state_history -> it returns the sequence of state snapshots saved after each node execution
print(list(workflow.get_state_history(config1)))
print(list(workflow.get_state_history(config2)))

