from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


class QE_State(TypedDict):
    a: int
    b: int
    c: int

    equation: str
    discriminent: float
    result: str


graph = StateGraph(QE_State)


def show_eq(state: QE_State) -> QE_State:
    eq = f"{state['a']}x^2 + {state['b']}x + {state['c']}"

    return {'equation': eq}

def calc_discri(state: QE_State) -> QE_State:
    d = state['b']**2 - (4*state['a']*state['c'])

    return {'discriminent': d}

def real_roots(state: QE_State) -> QE_State:
    root1 = (-state['b'] + (state['discriminent'])**0.5)/2*state['a']
    root2 = (-state['b'] - (state['discriminent'])**0.5)/2*state['a']

    result = f'd>0 and roots are real: {root1} and {root2}'

    return {'result': result}

def non_real_roots(state: QE_State) -> QE_State:
    return {'result': 'd<0 and roots are imaginary'}

def repeating_roots(state: QE_State) -> QE_State:
    root = (-state['b']) / (2 * state['a'])

    result = f'd = 0 and roots are repeating: {root}'

    return {'result': result}

# not a node function
def condition_checker(state: QE_State) -> Literal['real_roots', 'non_real_roots', 'repeating_roots']:
    if state['discriminent']>0:
        return 'real_roots'
    elif state['discriminent']<0:
        return 'non_real_roots'
    else:
        return 'repeating_roots'


graph.add_node('show_eq', show_eq)
graph.add_node('calc_discri', calc_discri)
graph.add_node('real_roots', real_roots)
graph.add_node('non_real_roots', non_real_roots)
graph.add_node('repeating_roots', repeating_roots)


graph.add_edge(START, 'show_eq')
graph.add_edge('show_eq', 'calc_discri')

graph.add_conditional_edges('calc_discri', condition_checker)

graph.add_edge('real_roots', END)
graph.add_edge('non_real_roots', END)
graph.add_edge('repeating_roots', END)


# complie graph
workflow = graph.compile()


initial_state = {
    'a': 1,
    'b': 2,
    'c': 1
}
final_state = workflow.invoke(initial_state)
print(final_state)


# visualize grpah
print(workflow.get_graph().print_ascii())
