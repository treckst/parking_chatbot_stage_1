from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState
    )
from langchain_classic.retrievers import EnsembleRetriever
from langgraph.graph.message import MessagesState
from datetime import datetime
from dotenv import load_dotenv
import json
from pathlib import Path
import sqlite3
import time
import json
from langgraph.prebuilt import ToolNode

load_dotenv()
from src.models_vector import llm, get_vector, response_llm
from src.schema import (
    MainState,
    UpgradedQuerySchema,
    GradeDocs,
    UserState,
    UserInfoExtraction,
    IntentSchema
)
from src.guardrails import InputSanitizer, SQLValidationError, safe_sql
from src.security import SecutiryGuard
from src.functions import get_latest_human_message
from src.functions import (
    dense_retriever,
    sparse_retriever,
)
from src.prompts import (
    QUESTION_REWRITER_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    SECUTIRY_ALERT_PROMPT,
    INTENTION_PROMPT,
    OPTIMIZER_PROMPT,
    EXTRACT_STATE_PROMPT,
    RESERVATION_REPLY_PROMPT,
    RETRIEVAL_PROMPT,
    SQL_AGENT_PROMPT,
)

DB_URI = (
    (Path(__file__).parent.parent / "data" / "parking.db")
    .resolve()
    .as_uri()
    + "?mode=ro"
)

MAX_RETRIES = 3
MAX_SQL_CALLS = 3

guard = SecutiryGuard()
sanitizer = InputSanitizer()

data_path = Path(__file__).parent.parent / 'data'
db_path = data_path / 'parking.db'
vector_store = get_vector(data_path)
semantic_search = dense_retriever(vector_store, 4)
keyword_search = sparse_retriever(vector_store, 4)
retriever = EnsembleRetriever(
    retrievers=[semantic_search, keyword_search],
    weights=[0.60, 0.40],  # 60% semantic importance, 40% keyword importance
    )

def input_guardrails(state: MainState):
    cleared_msg = sanitizer.sanitize(state.messages[-1].content)
    if sanitizer.is_suspicious(cleared_msg)[0] or not guard.llm_check(cleared_msg)['safe']:
        return 'security_stoppage'
    return 'message_retrieve'

# node which is the first to update state at every invocation, resetting sql calls with max retrieves
def message_retrieve(state):
    last_user_msg = get_latest_human_message(state.messages)
    current_clear_msg = sanitizer.sanitize(last_user_msg.content)
    messages = [
        SystemMessage(content=OPTIMIZER_PROMPT),
        HumanMessage(content=current_clear_msg)
    ]
    response = llm.with_structured_output(UpgradedQuerySchema).invoke(messages)
    query = response.finetuned_query
    return {
    "human_messages": [last_user_msg],
    "queries": [query],
    "sql_calls": 0,
    "max_retrieves": 0,
    "run_query": "",
}

def security_stoppage(state):
    human_msg = get_latest_human_message(state.messages)
    messages = [
            SystemMessage(content=SECUTIRY_ALERT_PROMPT),
            human_msg
        ]
    response = llm.invoke(messages)
    return {'messages': [response]}

# codnitioal edge, if irrelevant, should return the response to user after that, probably output schema or AIMessage and END
def check_intent(state):
    response = llm.with_structured_output(IntentSchema).invoke([
        SystemMessage(content=INTENTION_PROMPT),
        *[m for m in state.messages if isinstance(m, (HumanMessage, AIMessage))][-3:]
        ])
    if response.intention == 'reservation':
        return 'update_state'
    return 'docs_retrieval'

# after guardrails  has been passed and the query was saved to state, summarizing all messages 1: and removing them except the last two
def should_summarize(state):
    messages = state.messages
    # If there are more than 15 messages, then we summarize the conversation
    if len(messages) > 15:
        return "summarize_conversation"
    return 'intent_router'

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

    delete_messages = [RemoveMessage(id=m.id) for m in state.messages[:-4]]
    return {'summary': response.content, 'messages': delete_messages}

def update_state(state):

    extracted_info = llm.with_structured_output(UserInfoExtraction).invoke([
        SystemMessage(EXTRACT_STATE_PROMPT),
        HumanMessage(content=json.dumps({
            'current_message': get_latest_human_message(state.messages).content,
            'last_question': state.reservation_question,
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
            **existing_info["reservation_period"],
            **current_reservation_period
        }
    }
    user_info = UserState.model_validate(merged)

    return {"user_info": user_info}

def check_availability(state):
    user_info = state.user_info
    required_fields = {
        "your name": user_info.name,
        "your car registration number": user_info.car_number,
        "the parking place": user_info.place,
        "the arrival date and time": user_info.reservation_period.start_date,
        "the departure date and time": user_info.reservation_period.end_date,
    }
    missing = [field for field, value in required_fields.items() if not (value or "").strip()]
    period = []
    correction = ""
    for label in ("the arrival date and time", "the departure date and time"):
        value = required_fields[label]
        if label in missing:
            continue
        try:
            if len(value.strip()) <= 10:
                raise ValueError
            moment = datetime.fromisoformat(value.strip())
            if moment.tzinfo is not None:
                raise ValueError
        except ValueError:
            correction = "Please provide " + label + " in Warsaw local time, including the date."
            break
        if (moment.year, moment.month) != (2026, 11):
            correction = "Reservations cover 1–30 November 2026. What November date and times would you like?"
            break
        period.append(moment)
    if len(period) == 2:
        start, end = period
        if end.date() != start.date() or end <= start:
            correction = "Please choose a departure after arrival on the same day."
    question = correction
    if missing:
        question = (question + " Please provide " + ", ".join(missing) + ".").strip()
    return {"missing_info": missing, "reservation_question": question}

def route_after_availability(state):
    return "reservation_response" if state.reservation_question else "agent"


def authorize_sql(action, arg1, arg2, database, source):
    if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION}:
        return sqlite3.SQLITE_OK

    if action == sqlite3.SQLITE_READ:
        if database == "main" and arg1 == "parking_slots":
            return sqlite3.SQLITE_OK

    return sqlite3.SQLITE_DENY

@tool
def execute_readonly_sql(sql: str) -> dict:
    """Execute one read-only SQLite query against the public parking table.

    Returns either rows or a database error that can help correct the SQL.
    """
    connection = None

    try:
        sql = safe_sql(sql)
        connection = sqlite3.connect(DB_URI, uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        connection.set_authorizer(authorize_sql)
        deadline = time.monotonic() + 2
        connection.set_progress_handler(
            lambda: time.monotonic() > deadline, 1000,
        )
        rows = connection.execute(sql).fetchmany(51)

        return {
            "ok": True,
            "rows": [dict(row) for row in rows[:50]],
            "truncated": len(rows) > 50,
        }
    
    except SQLValidationError as e:
        return (f"Query rejected: {e}")

    except sqlite3.Error as exc:
        message = str(exc)

        if "no such column" in message:
            category = "unknown_column"
        elif "no such table" in message:
            category = "unknown_table"
        elif "syntax error" in message:
            category = "syntax_error"
        else:
            # Avoid exposing arbitrary operational error details.
            return {
                "ok": False,
                "error_category": "execution_error",
                "error": "Query was denied, timed out, or could not execute.",
                "retryable": False,
            }
        return {
                "ok": False,
                "error_category": category,
                "error": message,
                "retryable": True,
            }

    finally:
        if connection is not None:
            connection.close()


model_with_tools = llm.bind_tools(
    [execute_readonly_sql],
    parallel_tool_calls=False,
)

def agent(state):
    instructions = SQL_AGENT_PROMPT
    
    if state.sql_calls >= MAX_SQL_CALLS:
        instructions += """
No tool calls remain. Answer using successful results already obtained.
If no adequate result exists, explain that the lookup could not be completed.
"""

    # Remove tool access when the budget is exhausted.
    model = llm if state.sql_calls >= MAX_SQL_CALLS else model_with_tools
    if state.sql_calls == 0:
        model = model_with_tools.bind(tool_choice="execute_readonly_sql")
    request = state.user_info.model_dump(include={"place", "reservation_period"})
    turn_start = max(i for i, msg in enumerate(state.messages) if isinstance(msg, HumanMessage))
    response = model.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=json.dumps(request, ensure_ascii=False)),
        *state.messages[turn_start + 1:],
    ])
    return {"messages": [response]}

tool_node = ToolNode([execute_readonly_sql])

def tools(state):
    calls = state.messages[-1].tool_calls
    remaining = MAX_SQL_CALLS - state.sql_calls

    if len(calls) > remaining:
        return {
            "messages": [ToolMessage(
                    content=json.dumps({
                        "ok": False,
                        "error": "SQL call budget exceeded. No query executed.",
                        "retryable": False}),
                    tool_call_id=call["id"]) for call in calls],
            "sql_calls": MAX_SQL_CALLS,
        }

    result = tool_node.invoke(state)

    return {
        "messages": result["messages"],
        "sql_calls": state.sql_calls + len(calls),
    }

def route_after_agent(state):
    if state.messages[-1].tool_calls:
        return "tools"
    return 'reservation_response'

def reservation_response(state):
    if state.reservation_question:
        return {"messages": [AIMessage(content=state.reservation_question)]}
    
    prompt_template = ChatPromptTemplate.from_template(RESERVATION_REPLY_PROMPT)
    rag_chain = prompt_template | response_llm | StrOutputParser()

#last message here is resulted message from agent after calling sql and observing the data
    response = rag_chain.invoke({
        'context': state.messages[-1].content,
        'query': state.human_messages[-1].content,
        'user_info': state.user_info.model_dump(),
        'missing_info': state.missing_info})

    return {'messages': [AIMessage(content=response)]}


def docs_retrieval(state):
    if state.run_query:
        query = state.run_query
    else:
        query = state.queries[-1]

    result = retriever.invoke(query)
    return {'relevant_documents': [doc.page_content for doc in result]}

def grade_docs(state):
# conditional edge to check amount of retrieved docs, if 0, retry to format query and retry the retrieval and then ask for new query from user
    if state.run_query:
        query = state.run_query
    else:
        query = state.queries[-1]

    docs = state.relevant_documents

    structured_llm_grader = llm.with_structured_output(GradeDocs)
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', GRADE_DOCUMENTS_PROMPT),
            ('human', "Retrieved document: \n\n {document} \n\n User question: {question}")
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    filtered_docs = []
    for doc in docs:
        grader_response = retrieval_grader.invoke({'question': query, 'document': doc})
        if grader_response.binary_score.lower() == 'yes':
            filtered_docs.append(doc)

    return {'relevant_documents': filtered_docs, 'max_retrieves': state.max_retrieves + 1}

def response_or_fallback(state):
    if len(state.relevant_documents) > 0 or state.max_retrieves >= MAX_RETRIES:
        return 'retrieval_response'
    return 'transform_query'


def transform_query(state):
    improved_run_query = llm.invoke([
    SystemMessage(content=QUESTION_REWRITER_PROMPT),
    HumanMessage(content=f"Initial question: {state.queries[-1]}")
        ])
    
    return {'run_query': improved_run_query.content}

def retrieval_response(state):
    if not state.relevant_documents:
        response = llm.invoke([
            SystemMessage(content="Answer with I couldn’t find that in the parking information"),
            HumanMessage(content=get_latest_human_message(state.messages).content)
            ]) 
    else:
        prompt_template = ChatPromptTemplate.from_template(RETRIEVAL_PROMPT)
        retrieval_chain = prompt_template | response_llm | StrOutputParser()
        response = retrieval_chain.invoke({
            'question': get_latest_human_message(state.messages).content,
            'context' : state.relevant_documents})

    return {'messages': [AIMessage(content=response)]}


# add to each ending the last check with nlp model
workflow = StateGraph(MainState, input_schema=MessagesState, output_schema=MessagesState)
workflow.add_node(security_stoppage)
workflow.add_conditional_edges(START, input_guardrails, ['security_stoppage', 'message_retrieve'])
workflow.add_node(message_retrieve)
workflow.add_edge('security_stoppage', END)
workflow.add_conditional_edges('message_retrieve',should_summarize, ['summarize_conversation', 'intent_router'])
workflow.add_node(summarize_conversation)
workflow.add_node(update_state)
workflow.add_node(check_availability)
workflow.add_node("intent_router", lambda state: {})
workflow.add_node(docs_retrieval)
workflow.add_node(agent)
workflow.add_node(tools)
workflow.add_node(grade_docs)
workflow.add_node(transform_query)
workflow.add_node(retrieval_response)
workflow.add_edge('retrieval_response', END)
workflow.add_conditional_edges('summarize_conversation', check_intent, ['docs_retrieval', 'update_state'] )
workflow.add_conditional_edges('intent_router', check_intent, ['docs_retrieval', 'update_state'] )
workflow.add_edge('docs_retrieval', 'grade_docs')
workflow.add_conditional_edges('grade_docs', response_or_fallback, ['retrieval_response', 'transform_query'])
workflow.add_edge('transform_query', 'docs_retrieval')
workflow.add_edge('update_state', 'check_availability')
workflow.add_conditional_edges('check_availability', route_after_availability, ['agent', 'reservation_response'])
workflow.add_conditional_edges("agent", route_after_agent, ['tools', 'reservation_response'])
workflow.add_node(reservation_response)
workflow.add_edge("tools", "agent")
workflow.add_edge('reservation_response', END)
graph = workflow.compile(checkpointer=InMemorySaver())
