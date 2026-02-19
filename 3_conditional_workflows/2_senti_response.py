from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')


model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    api_key=api_key
)


class Sentiment_Schema(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='tell the sentiment of the review is positive or negative')
struct_model1 = model.with_structured_output(Sentiment_Schema)

class Diagnosis_Schema(BaseModel):
    main_issue: Literal['UI/UX', 'Payments', 'Bug', 'Others'] = Field(description='tell the main issue user is facing')
    mood: Literal['Angry', 'Calm', 'Fraustated'] = Field(description='tell the mood of the use')
    urgency: Literal['low', 'medium', 'high'] = Field(description='tell how urgent the user issue should be fixed')
struct_model2 = model.with_structured_output(Diagnosis_Schema)

# main state
class Main_State(TypedDict):
    review: str

    sentiment: str
    diagnosis: dict
    response: str


graph = StateGraph(Main_State)


def sentiment(state: Main_State) -> Main_State:
    prompt = f'Tell the sentiment of this review is either positive or negative? \n {state["review"]}'

    response = struct_model1.invoke(prompt)

    return {'sentiment': response.sentiment}

def positive_response(state: Main_State) -> Main_State:
    prompt = f'Write a good response to this positive review and at the end ask user to also give feedback \n {state["review"]}'

    response = model.invoke(prompt).content

    return {'response': response}

def diagnosis(state: Main_State) -> Main_State:
    prompt = f'According to this review tell the main_issue and urgency of this negative review and also mood of the user \n {state["review"]}'

    response = struct_model2.invoke(prompt)

    return {'diagnosis': response.model_dump()}

def negative_response(state: Main_State) -> Main_State:
    prompt = f'Write a  response to this negative review according to these {state["diagnosis"]} and be respectful with user \n {state["review"]}'

    response = model.invoke(prompt).content

    return {'response': response}

def condition_checker(state: Main_State) -> Literal['positive_response', 'diagnosis']:
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'diagnosis'


# define nodes
graph.add_node('sentiment', sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('diagnosis', diagnosis)
graph.add_node('negative_response', negative_response)


# define edges
graph.add_edge(START, 'sentiment')

graph.add_conditional_edges('sentiment', condition_checker)

graph.add_edge('positive_response', END)
graph.add_edge('diagnosis', 'negative_response')
graph.add_edge('negative_response', END)


# complie graph
workflow = graph.compile()


initial_state = {
    'review': 'This website is is very good, very simple to use and so lite loved it'
}
final_state = workflow.invoke(initial_state)
print(final_state)


# visualize grpah
print(workflow.get_graph().print_ascii())
