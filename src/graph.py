from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState
    )
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv()
from src.models_vector import llm, answer_llm
from src.schema import (
    MainState,
    InputSchema,
    UpgradedQuerySchema,
    RefinedQuerySchema,
    GradeDocs,
    UserState,
    UserInfoExtraction
)
from src.guardrails import InputSanitizer
from src.security import SecutiryGuard
from src.functions import get_latest_human_message
from src.react_graph import sql_agent
from src.retrieval_graph import retrieval_graph

SECUTIRY_ALERT_PROMPT = """You are assistant for security against the injection prompts to bypass the guardrails of parking reservation system.
You should write the response to the user who has used the injecton:"""

IRRELEVANCY_ALERT_PROMPT = """You are a chat bot assistant for parking reservation system.
Respond to the user that his query has nothing to do with parking reservation and it's irrelevant, ask for another query
User query:\n"""

OPTIMIZER_PROMPT = """You are professional optimization assistant who rebuild queries for hybrid search.
Output upgraded query for semantic and key-word search at the same time.
Strip irrelevant info which dont contribute to the information that user intents to get"""

RELEVANCE_PROMPT = """Check if the current query is relevant to the parking reservation.
While also taking into account the user info and missing info which is required to be collected/updated if user provides anything related.
Current state: (can be empty) \n\n{user_profile}
Missing fields: (if any) \n\n{missing_info}"""

EXTRACT_STATE_PROMPT = """Extract reservation details supplied or corrected in the latest user message.
Use null for fields the user did not provide"""
#connect to sql and chroma as runtime (outside of graph) 
# update state with returning the missing info always
# get input - guardrails on input - llm to update its user's state if possible, fine-tune the query, 
# retrieve sql and hybrid search unsemble in parallel,
# guard rails on query in sql for only select and quardrails on output from sql with nlp model
# guard rails on retrieved documents with nlp model
# retrieve the vector_db based on query if it was relevant and fine-tune the query for search, and store in contextstate the documents,
# now decide if missing some info in state so mention to agent what is missing and ask user + response based on context and original query.
# iterate over until the user's state if enought for reservation,
# invoke next graph with human-in-the-loop with administator to approve the reservation
# invoke next graph and based on admin response - implement reservation and answer the user or if rejected, response as well to user, maybe with admin message why.

guard = SecutiryGuard()
sanitizer = InputSanitizer()


def input_guardrails(state):
    cleared_msg = sanitizer.sanitize(state.messages[-1].content)
    if sanitizer.is_suspicious(cleared_msg)[0] or not guard.llm_check(cleared_msg):
        return 'security_stoppage'
    return 'check_relevance'

def security_stoppage(state):
    human_msg = get_latest_human_message(state.messages)
    messages = [
            SystemMessage(content=SECUTIRY_ALERT_PROMPT),
            human_msg
        ]
    response = llm.invoke(messages)
    return {'messages': [response]}

def irrelevant_query(state):
    human_msg = get_latest_human_message(state.messages)
    messages = [
        SystemMessage(content=IRRELEVANCY_ALERT_PROMPT),
        human_msg
    ]
    response = llm.invoke(messages)
    return {'messages': [response]}

# codnitioal edge, if irrelevant, should return the response to user after that, probably output schema or AIMessage and END
def check_relevance(state):
    human_msg = get_latest_human_message(state.messages)
    messages = RELEVANCE_PROMPT.format(user_profile=state.user_info.model_dump(), missing_info=state.missing_info)
    response = llm.with_structured_output(RefinedQuerySchema).invoke([
        SystemMessage(content=messages),
        human_msg
        ])
    if response.query_relevant == 'relevant':
        return 'query_optimization'
    return 'irrelevant_query'

def query_optimization(state):
    human_msg = get_latest_human_message(state.messages)
    current_clear_msg = sanitizer.sanitize(human_msg.content)
    messages = [
        SystemMessage(content=OPTIMIZER_PROMPT),
        human_msg
    ]
    response = llm.with_structured_output(UpgradedQuerySchema).invoke(messages)
    query = response.finetuned_query
    return {'current_message' : current_clear_msg, 'queries': [query]}

# after guardrails  has been passed and the query was saved to state, summarizing all messages 1: and removing them except the last two
def should_summarize(state):
    messages = state.messages
    # If there are more than 15 messages, then we summarize the conversation
    if len(messages) > 15:
        return "summarize_conversation"
    return 'react'

def summarize_conversation(state: MainState):
    if state.summary:
        summary_message = (
            f"This is summary of the conversation to date: {state.summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
        messages = state.messages + [HumanMessage(summary_message)]
        response = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state.messages[:-2]]
    return {'summary': response.content, 'messages': delete_messages}

def should_update_state(state):
    sys_msg = 'Decide either its appropriate to do any state update or if its possible'

# check later if there would be a need to supply summary with all messages history, also depending on the final output schema
def update_state(state):

    extracted_info = llm.with_structured_output(UserInfoExtraction).invoke([
        SystemMessage(EXTRACT_STATE_PROMPT),
        HumanMessage(content=json.dumps({
            'current_message': state.current_message,
            'existing_user_info': state.user_info.model_dump(mode="json"),
            'missing_fields': state.missing_info
        }))
        ])
    existing_info = state.user_info.model_dump()
    current_info = extracted_info.model_dump(exclude_none=True)
    current_reservation_period = current_info.pop('reservation_period', {})

    merged = {
        **existing_info,
        **current_info,
        "reservation_period" : {
            **existing_info["reservation_period"]
            **current_reservation_period
        }
    }
    user_info = UserState.model_validate(merged)

    required_fields = {
        "name": user_info.name,
        "car_number": user_info.car_number,
        "reservation_period.start_end": user_info.reservation_period.start_date,
        'reservation_period.end_date': user_info.reservation_period.end_date
    }

    missing_info = [
        field for field, value in required_fields.items()
        if value is None or (
            isinstance(value, str) and not value.strip()
        )
    ]   

    return {"user_info": user_info, "missing_info": missing_info}

# do parent-chunking, than implement ensemble retrieval, semantic 0.6, within node do guardrails before saving to state
# check if each document is relevant, and only then write it down
def generate_response(state):
    #gives summary, all messages with summary and docs, generate the response
    rag_prompt = """You are an assistant for Parking Space Reservation.
    You are collecting user info to start the reservation.
    It must match the possible dates, and there must be no conflicts or inaccuracies in the information.
    Use the following pieces of retrieved context to answer the query.
    Make sure to ask user for missing info for the reservation
    Keep the answer concise.

Current summary: {summary}

User query: {question} 

Current user info (if any): {user_info}

Currently missing information for the reservation(if any): {missing_information}

Context: {context} 

Answer:"""
    response = answer_llm.invoke()
# add state update implementation
#probably another graph inside with ToolCalling or structured output

workflow = StateGraph(MainState, inputschema=InputSchema)
workflow.add_node(security_stoppage)
workflow.add_node("relevance_router", lambda state: {})
workflow.add_conditional_edges(START, input_guardrails, {
    'security_stoppage': 'security_stoppage',
    'check_relevance' : 'relevance_router'
    })
workflow.add_conditional_edges('relevance_router', check_relevance, ['query_optimization', 'irrelevant_query'] )
workflow.add_node(query_optimization)
workflow.add_node(irrelevant_query)
workflow.add_edge('security_stoppage', END)
workflow.add_edge('irrelevant_query', END)
workflow.add_node(query_optimization)
workflow.add_node(summarize_conversation)
workflow.add_node(update_state)
workflow.add_node('react', sql_agent)
workflow.add_node('retrieval', retrieval_graph)
workflow.add_node(generate_response)
workflow.add_conditional_edges('query_optimization', should_summarize, ['summarize_conversation', 'react'])
workflow.add_edge('summarize_conversation', 'react')
workflow.add_edge('react', 'update_state')
workflow.add_edge('update_state', 'generate_response')
workflow.add_edge('generate_response', END)
# react should run in parallel with retrieval from rag and self-fixing query and grading, probably another subgraph