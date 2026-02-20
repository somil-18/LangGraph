from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator
import os


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')


# define llm models
generator_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)
eval_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)
optimize_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=api_key)

class eval_schema(BaseModel):
    status: Literal['approved', 'needs_improvements'] = Field(description='status approved or needs_improvements based on feedback')
    feedback: str = Field(description='feedback of the tweet')
struct_eval_model = eval_model.with_structured_output(eval_schema)


# define state
class Tweet_State(TypedDict):
    topic: str
    max_iteration: int

    tweet: str
    feedback: str
    status: Literal['approved', 'needs_improvements']
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

    iteration: int


# define graph
graph = StateGraph(Tweet_State)


# function
def generate_llm(state: Tweet_State) -> Tweet_State:
    prompt = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]

    response = generator_model.invoke(prompt).content

    return {'tweet': response, 'tweet_history': [response]}

def eval_llm(state: Tweet_State) -> Tweet_State:
    prompt = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality - Is this fresh, or have you seen it a hundred times before?  
2. Humor - Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness - Is it short, sharp, and scroll-stopping?  
4. Virality Potential - Would people retweet or share it?  
5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- status: "approved" or "needs_improvements"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
    ]

    response = struct_eval_model.invoke(prompt)

    return {'feedback': response.feedback, 'status': response.status, 'feedback_history': [response.feedback]}

def optimizer_llm(state: Tweet_State) -> Tweet_State:
    prompt = [ 
    SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
    HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]
    
    response = optimize_model.invoke(prompt).content

    iter = state['iteration'] + 1

    return {'tweet': response, 'iteration': iter, 'tweet_history':[response]}

def condition_checker(state: Tweet_State):
    if state['status'] == 'approved' or state['iteration']>=state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvements'


# define nodes
graph.add_node('generate', generate_llm)
graph.add_node('evaluate', eval_llm)
graph.add_node('optimize', optimizer_llm)


# define edges
graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')

graph.add_conditional_edges('evaluate', condition_checker, {'approved': END, 'needs_improvements': 'optimize'})

graph.add_edge('optimize', 'evaluate')


# complie graph
workflow = graph.compile()


initial_state = {
    'topic': 'India Civic Sense',
    'iteration': 0,
    'max_iteration': 4
}
final_state = workflow.invoke(initial_state)
print(final_state)


# visualize the graph
print(workflow.get_graph().print_ascii())
