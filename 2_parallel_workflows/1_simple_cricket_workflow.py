from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# define state
class CricketState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int 

    sr: float
    bpb: float
    boundary_perct: float
    summary: str


# define your graph
graph = StateGraph(CricketState)


# define functions
def calculate_sr(state: CricketState) -> CricketState:
    sr = (state['runs']/state['balls'])*100

    # return state # WRONG
    return {'sr': sr}

def calculate_bpb(state: CricketState) -> CricketState:
    bpb = state['balls']/(state['fours']+state['sixes'])

    # return state # WRONG
    return {'bpb': bpb}

def calculate_boundary_perct(state: CricketState) -> CricketState:
    bp = (((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs']) * 100

    # return state # WRONG
    return {'boundary_perct': bp}

def summary(state: CricketState) -> CricketState:
    state['summary'] = f'Total runs scored: {state["runs"]} \n Total balls played: {state["balls"]} \n Total fours: {state["fours"]} \n Total sixes: {state["sixes"]} \n\n Strike Rate: {state["sr"]} \n Boundary per Balls: {state["bpb"]} \n Boundary percentage: {state["boundary_perct"]} \n\n Thank You'

    return state


# define nodes
graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_perct', calculate_boundary_perct)
graph.add_node('summary', summary)


# define edges
graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_perct')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_perct', 'summary')

graph.add_edge('summary', END)


# complie graph
workflow = graph.compile()


# execute graph
initial_state = {
    'runs': 101,
    'balls': 69,
    'fours': 8,
    'sixes': 4,
    'sr': 0,
    'bpb': 0,
    'boundary_perct': 0,
    'summary': ''
}
final_state = workflow.invoke(initial_state)
print(final_state)


# visualize grpah
print(workflow.get_graph().print_ascii())


'''
1) In LangGraph parallel workflows, every node receives the entire shared state.
2) But each node must perform a partial update - it should only write the fields it is responsible for.
3) This is because parallel branches run independently, and if two nodes rewrite the whole state, the last one to finish will overwrite the others!
4) So with partial update, LangGraph can merge the result from all branches
5) The rule is - read everything, write only what you own
'''
