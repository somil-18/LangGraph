from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# define state
class BMI_State(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str


# define function
def calculate_bmi(state: BMI_State) -> BMI_State:
    weight = state['weight_kg']
    height =state['height_m']

    bmi = weight/(height**2)

    state['bmi'] = round(bmi, 2)

    return state

def category(state: BMI_State) -> BMI_State:
    bmi = state['bmi']

    if bmi<18.5:
        state['category'] = 'Underweight'
    elif 18.5<=bmi<25:
        state['category'] = 'Normal'
    elif 25<=bmi<30:
        state['category'] = 'Overweight'
    else:
        state['category'] = 'Obese'

    return state


# define your graph
graph = StateGraph(BMI_State)


# define nodes of your graph
graph.add_node('calculate_bmi', calculate_bmi)
graph.add_node('category', category)


# define edges of your graph
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'category')
graph.add_edge('category', END)


# complie the graph
workflow = graph.compile()


# execute the graph
initial_state = {'weight_kg': 67, 'height_m': 1.75}
final_state = workflow.invoke(initial_state)
print(final_state)


# visualize graph
print(workflow.get_graph().print_ascii())

