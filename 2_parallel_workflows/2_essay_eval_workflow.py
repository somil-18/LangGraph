from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import operator


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# llm model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
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
    cot_feedback: str # clarity of thought
    doa_feedback: str # depth of analysis
    g_feedback: str # grammer
    summarized_feedback: str
    scores: Annotated[list[int], operator.add]
    avg_score: float


# define functions
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

#  merge barrier
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


# define graph
graph = StateGraph(Essay_state)


# graph nodes
graph.add_node("cot", cot)
graph.add_node("doa", doa)
graph.add_node("g", g)
graph.add_node("merge_barrier", merge)
graph.add_node("final_eval", final_eval)


# graph edges
graph.add_edge(START, "cot")
graph.add_edge(START, "doa")
graph.add_edge(START, "g")
graph.add_edge("cot", "merge_barrier")
graph.add_edge("doa", "merge_barrier")
graph.add_edge("g", "merge_barrier")
graph.add_edge("merge_barrier", "final_eval")
graph.add_edge("final_eval", END)


# compile the graph
workflow = graph.compile()


# test the graph
essay = '''
India, officially the Republic of India,[j][20] is a country in South Asia. It is the seventh-largest country by area; the most populous country since 2023;[21] and, since its independence in 1947, the world's most populous democracy.[22][23][24] Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[k] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Myanmar, Thailand, and Indonesia.

Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago.[26][27][28] Their long occupation, predominantly in isolation as hunter-gatherers, has made the region highly diverse.[29] Settled life emerged on the subcontinent in the western margins of the Indus river basin 9,000 years ago, evolving gradually into the Indus Valley Civilisation of the third millennium BCE.[30] By 1200 BCE, an archaic form of Sanskrit, an Indo-European language, had diffused into India from the northwest.[31][32] Its hymns recorded the early dawnings of Hinduism in India.[33] India's pre-existing Dravidian languages were supplanted in the northern regions.[34] By 400 BCE, caste had emerged within Hinduism,[35] and Buddhism and Jainism had arisen, proclaiming social orders unlinked to heredity.[36] Early political consolidations gave rise to the loose-knit Maurya and Gupta Empires.[37] This era was noted for creativity in art, architecture, and writing,[38] but the status of women declined,[39] and untouchability became an organised belief.[l][40] In South India, the Middle kingdoms exported Dravidian language scripts and religious cultures to the kingdoms of Southeast Asia.[41]

In the 1st millennium, Islam, Christianity, Judaism, and Zoroastrianism became established on India's southern and western coasts.[42] In the early centuries of the 2nd millennium Muslim armies from Central Asia intermittently overran India's northern plains.[43] The resulting Delhi Sultanate drew northern India into the cosmopolitan networks of medieval Islam.[44] In south India, the Vijayanagara Empire created a long-lasting composite Hindu culture.[45] In the Punjab, Sikhism emerged, rejecting institutionalised religion.[46] The Mughal Empire ushered in two centuries of economic expansion and relative peace,[47] and left a rich architectural legacy.[48][49] Gradually expanding rule of the British East India Company turned India into a colonial economy but consolidated its sovereignty.[50] British Crown rule began in 1858. The rights promised to Indians were granted slowly,[51][52] but technological changes were introduced, and modern ideas of education and the public life took root.[53] A nationalist movement emerged in India, the first in the non-European British Empire and an influence on other nationalist movements.[54][55] Noted for nonviolent resistance after 1920,[56] it became the primary factor in ending British rule.[57] In 1947, the British Indian Empire was partitioned into two independent dominions, a Hindu-majority dominion of India and a Muslim-majority dominion of Pakistan. A large-scale loss of life and an unprecedented migration accompanied the partition.[58]

India has been a federal republic since 1950, governed through a democratic parliamentary system. It is a pluralistic, multilingual and multi-ethnic society. India's population grew from 361 million in 1951 to over 1.4 billion in 2023.[59] During this time, its nominal per capita income increased from US$64 annually to US$2,601, and its literacy rate from 16.6% to 74%. A comparatively destitute country in 1951,[60] India has become a fast-growing major economy and a hub for information technology services, with an expanding middle class.[61] India has reduced its poverty rate, though at the cost of increasing economic inequality.[62] It is a nuclear-weapon state that ranks high in military expenditure. It has disputes over Kashmir with its neighbours, Pakistan and China, unresolved since the mid-20th century.[63] Among the socio-economic challenges India faces are gender inequality, child malnutrition,[64] and rising levels of air pollution.[65] India's land is megadiverse with four biodiversity hotspots. India's wildlife, which has traditionally been viewed with tolerance in its culture,[66] is supported in protected habitats.
'''

initial_state = {
    "essay": essay,
    "cot_feedback": "",
    "doa_feedback": "",
    "g_feedback": "",
    "summarized_feedback": "",
    "scores": [],
    "avg_score": 0.0
}

final_state = workflow.invoke(initial_state)

print(final_state)


# visualize graph
print(workflow.get_graph().print_ascii())


'''
What about 'merge_barrier' node?

- It is a synchronization node in parallel workflows
- It's purpose is to ensure that all parallel branches finish execution before the graph continues to the next node
- Without a merge barrier, the downstream node may execute as soon as the first parallel branch finishes, which can cause it to run with incomplete or partial state.
- We are only returning empty dictionary, if we returned a full state then LangGraph would think new data is being added and might apply reducer again, causing duplicate values ---> [8, 7, 9, 8, 7, 9]
'''