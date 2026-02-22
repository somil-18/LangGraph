from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
import time


# define the state
class CrashState(TypedDict):
    input: str
    step1: str
    step2: str


# define functions
def step_1(state: CrashState) -> CrashState:
    print("Step 1 executed")
    return {"step1": "done", "input": state["input"]}

def step_2(state: CrashState) -> CrashState:
    print("Step 2 hanging... now manually interrupt from the notebook toolbar (STOP button)")
    time.sleep(10)  
    return {"step2": "done"}

def step_3(state: CrashState) -> CrashState:
    print("Step 3 executed")
    return {"done": True}


graph = StateGraph(CrashState)
graph.add_node("step_1", step_1)
graph.add_node("step_2", step_2)
graph.add_node("step_3", step_3)


graph.set_entry_point("step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", "step_3")
graph.add_edge("step_3", END)


checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)


try:
    print("Running graph: Please manually interrupt during Step 2...")
    workflow.invoke({"input": "start"}, config={"configurable": {"thread_id": 'thread-1'}})
except KeyboardInterrupt:
    print("Kernel manually interrupted (crash simulated).")


# re-run to show fault-tolerant resume
print("\nRe-running the graph to demonstrate fault tolerance...")
final_state = workflow.invoke(None, config={"configurable": {"thread_id": 'thread-1'}})
print("\nFinal State:", final_state)


# state_history
print(list(workflow.get_state_history({"configurable": {"thread_id": 'thread-1'}}))) 


