from datetime import date, datetime, time as dt_time
import json
from pathlib import Path
import sqlite3
import time

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

from src.functions import dense_retriever, get_latest_human_message, sparse_retriever
from src.guardrails import InputSanitizer, SQLValidationError, safe_sql
from src.models_vector import get_vector, llm, response_llm
from src.prompts import (
    EXTRACT_STATE_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    INTENTION_PROMPT,
    OPTIMIZER_PROMPT,
    QUESTION_REWRITER_PROMPT,
    RESERVATION_REPLY_PROMPT,
    RETRIEVAL_PROMPT,
    SECUTIRY_ALERT_PROMPT,
    SQL_AGENT_PROMPT,
)
from src.schema import (
    GradeDocs,
    IntentSchema,
    MainState,
    UpgradedQuerySchema,
    UserInfoExtraction,
    UserState,
)
from src.security import SecutiryGuard

MAX_RETRIES = 3
MAX_SQL_CALLS = 3

DATA_PATH = Path(__file__).parent.parent / "data"
DB_URI = (DATA_PATH / "parking.db").resolve().as_uri() + "?mode=ro"

guard = SecutiryGuard()
sanitizer = InputSanitizer()

vector_store = get_vector(DATA_PATH)
retriever = EnsembleRetriever(
    retrievers=[dense_retriever(vector_store, 4), sparse_retriever(vector_store, 4)],
    weights=[0.60, 0.40],  # 60% semantic importance, 40% keyword importance
)

intent_llm = llm.with_structured_output(IntentSchema)
optimizer_llm = llm.with_structured_output(UpgradedQuerySchema)
extractor_llm = llm.with_structured_output(UserInfoExtraction)

retrieval_grader = (
    ChatPromptTemplate.from_messages([
        ("system", GRADE_DOCUMENTS_PROMPT),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    | llm.with_structured_output(GradeDocs)
)

retrieval_chain = (
    ChatPromptTemplate.from_template(
        RETRIEVAL_PROMPT + "\nAnswer the static parking question only."
    )
    | response_llm
    | StrOutputParser()
)

reservation_reply_chain = (
    ChatPromptTemplate.from_template(RESERVATION_REPLY_PROMPT)
    | response_llm
    | StrOutputParser()
)


def is_reviewing_reservation(state: MainState) -> bool:
    # true if currently user provided the message which wants reservation and no reservation_question (no missing fields and correct reservation period after last user_update)
    return state.wants_reservation and not state.reservation_question


def reply_with_static(state: MainState, *parts: str) -> dict:
    content = "\n\n".join(part for part in (state.static_answer, *parts) if part)
    return {"messages": [AIMessage(content=content)]}


def _join_items(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def validate_reservation(user_info: UserState) -> tuple[list[str], str]:
    missing: list[str] = []
    for label, value in (
        ("your name", user_info.name),
        ("your car registration number", user_info.car_number),
        ("the parking place", user_info.place),
    ):
        if not (value or "").strip():
            missing.append(label)

    period: list[datetime] = []
    correction = ""
    known_date: date | None = None
    needs_date = False

    date_fields = (
        ("arrival", user_info.reservation_period.start_date),
        ("departure", user_info.reservation_period.end_date),
    )
    for role, value in date_fields:
        raw = (value or "").strip()
        if not raw:
            missing.append(f"the {role} time" if known_date else f"the {role} date and time")
            continue

        # Date-only (e.g. "2026-11-27") — date was provided, only time is missing.
        if len(raw) <= 10:
            try:
                parsed_date = date.fromisoformat(raw)
            except ValueError:
                try:
                    dt_time.fromisoformat(raw)
                    needs_date = True
                    continue
                except ValueError:
                    missing.append(f"the {role} date and time")
                    continue

            if (parsed_date.year, parsed_date.month) != (2026, 11):
                correction = "Reservations cover 1–30 November 2026. What November date and times would you like?"
                break
            known_date = parsed_date
            missing.append(f"the {role} time")
            continue

        try:
            moment = datetime.fromisoformat(raw)
        except ValueError:
            missing.append(f"the {role} date and time")
            continue

        if moment.tzinfo is not None:
            correction = f"Please provide the {role} date and time in Warsaw local time (without a timezone offset)."
            break
        if (moment.year, moment.month) != (2026, 11):
            correction = "Reservations cover 1–30 November 2026. What November date and times would you like?"
            break
        known_date = moment.date()
        period.append(moment)

    if needs_date and not correction:
        missing.append("the parking date")

    if not correction and len(period) == 2:
        start, end = period
        if end.date() != start.date() or end <= start:
            correction = "Please choose a departure time after arrival on the same day."

    if correction and missing:
        question = f"{correction} Please also provide {_join_items(missing)}."
    elif correction:
        question = correction
    elif missing:
        question = f"Please provide {_join_items(missing)}."
    else:
        question = ""

    return missing, question


def route_entry(state: MainState):
    cleared_msg = sanitizer.sanitize(state.messages[-1].content)
    if sanitizer.is_suspicious(cleared_msg)[0] or not guard.llm_check(cleared_msg)["safe"]:
        return "security_stoppage"
    if len(state.messages) > 15:
        return "summarize_conversation"
    return "intent_router"


def security_stoppage(state: MainState):
    human_msg = get_latest_human_message(state.messages)
    response = llm.invoke([SystemMessage(content=SECUTIRY_ALERT_PROMPT), human_msg])
    return {"messages": [response]}


def summarize_conversation(state: MainState):
    if state.summary:
        summary_message = (
            f"This is summary of the conversation to date: {state.summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    response = llm.invoke(state.messages + [HumanMessage(content=summary_message)])
    delete_messages = [RemoveMessage(id=m.id) for m in state.messages[:-4]]
    return {"summary": response.content, "messages": delete_messages}


def check_intent(state: MainState):
    recent_messages = [
        msg for msg in state.messages if isinstance(msg, (HumanMessage, AIMessage))
    ][-3:]

    intent = intent_llm.invoke([
        SystemMessage(content=INTENTION_PROMPT.format(
            reservation_question=state.reservation_question or "None"
        )),
        *recent_messages,
    ])
    return {
        "wants_static_info": intent.static_info,
        "wants_dynamic_info": intent.dynamic_info,
        "wants_reservation": intent.reservation,
        "dynamic_question": intent.dynamic_question,
        "static_answer": "",
        "sql_calls": 0,
        "max_retrieves": 0,
        "run_query": "",
    }


def route_after_retrieval(state: MainState):
    if state.wants_reservation:
        return "update_state"
    if state.wants_dynamic_info:
        return "agent"
    return "info_response"


def route_after_intent(state: MainState):
    if state.wants_static_info:
        return "docs_retrieval"
    return route_after_retrieval(state)


def update_state(state: MainState):
    extracted_info = extractor_llm.invoke([
        SystemMessage(content=EXTRACT_STATE_PROMPT),
        HumanMessage(content=json.dumps({
            "current_message": get_latest_human_message(state.messages).content,
            "last_question": state.reservation_question,
            "existing_user_info": state.user_info.model_dump(mode="json"),
            "missing_fields": state.missing_info,
        })),
    ])
    existing_info = state.user_info.model_dump()
    current_info = extracted_info.model_dump(exclude_none=True)
    current_reservation_period = current_info.pop("reservation_period", {})

    merged = {
        **existing_info,
        **current_info,
        "reservation_period": {
            **existing_info["reservation_period"],
            **current_reservation_period,
        },
    }
    user_info = UserState.model_validate(merged)
    missing, question = validate_reservation(user_info)
    return {
        "user_info": user_info,
        "missing_info": missing,
        "reservation_question": question,
    }


def route_after_update(state: MainState):
    if not state.reservation_question or state.wants_dynamic_info:
        return "agent"
    return "reservation_response"


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

    except SQLValidationError as exc:
        return {
            "ok": False,
            "error_category": "validation_error",
            "error": f"Query rejected: {exc}",
            "retryable": True,
        }

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
model_with_required_tool = llm.bind_tools(
    [execute_readonly_sql],
    parallel_tool_calls=False,
    tool_choice="execute_readonly_sql",
)
tool_node = ToolNode([execute_readonly_sql])


def agent(state: MainState):
    reviewing_reservation = is_reviewing_reservation(state)

    if reviewing_reservation:
        instructions = SQL_AGENT_PROMPT + """
Review the supplied venue and booking period. Check that a slot is available
for the entire period. A reservation still requires approval.
If no available slot matches the requested venue or period (including if the
venue name returns no rows), query what parking venues and slots ARE available
on that parking_date (including location, opening hours, price per hour, and
number of available slots) and report them in your answer.
If an information question is supplied, answer it too. Reuse query results
when they cover both requests. Keep the venue and dates of each request distinct.
"""
        request = state.user_info.model_dump(include={"place", "reservation_period"})
        if state.wants_dynamic_info:
            request["information_question"] = state.dynamic_question
        question = json.dumps(request, ensure_ascii=False)
    else:
        instructions = SQL_AGENT_PROMPT + """
Answer the public parking question about hours, price, or availability.
If the requested venue or period has no available slots on a given date, also
query and report what parking venues, hours, and prices ARE available on that day.
If a venue or date is needed, ask for it. Do not ask for a name or plate,
start a reservation, or suggest that a reservation has been made.
"""
        question = state.dynamic_question

    if state.sql_calls >= MAX_SQL_CALLS:
        instructions += """
No tool calls remain. Answer using successful results already obtained.
If no adequate result exists, explain that the lookup could not be completed.
"""
        model = llm
    elif reviewing_reservation and state.sql_calls == 0:
        model = model_with_required_tool
    else:
        model = model_with_tools
    # finds the highest index of humanmessage from state.messages (grabs the last human message)
    turn_start = max(
        i for i, msg in enumerate(state.messages) if isinstance(msg, HumanMessage)
    )
    # on the first pass there is no message after human message passed
    # + 1: slices the list starting immediately after human message (the most recent ai messages with tool call)
    response = model.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=question),
        *state.messages[turn_start + 1:],
    ])

    if reviewing_reservation or response.tool_calls:
        return {"messages": [response]}

    booking_question = state.reservation_question if state.wants_reservation else ""
    return reply_with_static(state, response.text, booking_question)


def route_after_agent(state: MainState):
    if state.messages[-1].tool_calls:
        return "tools"
    if is_reviewing_reservation(state):
        return "reservation_response"
    return END


def info_response(state: MainState):
    fallback = (
        "" if state.static_answer
        else "I can answer parking questions or help you start a reservation."
    )
    return reply_with_static(state, fallback)


def tools(state: MainState):
    calls = state.messages[-1].tool_calls
    remaining = MAX_SQL_CALLS - state.sql_calls

    if len(calls) > remaining:
        return {
            "messages": [
                ToolMessage(
                    content=json.dumps({
                        "ok": False,
                        "error": "SQL call budget exceeded. No query executed.",
                        "retryable": False,
                    }),
                    tool_call_id=call["id"],
                )
                for call in calls
            ],
            "sql_calls": MAX_SQL_CALLS,
        }

    result = tool_node.invoke(state)
    return {
        "messages": result["messages"],
        "sql_calls": state.sql_calls + len(calls),
    }


def reservation_response(state: MainState):
    if state.reservation_question:
        return reply_with_static(state, state.reservation_question)

    # Last message here is the agent's response after running SQL checks.
    response = reservation_reply_chain.invoke({
        "context": state.messages[-1].content,
        "query": get_latest_human_message(state.messages).content,
        "user_info": state.user_info.model_dump(),
        "missing_info": state.missing_info,
    })
    return reply_with_static(state, response)


def docs_retrieval(state: MainState):
    query = state.run_query
    if not query:
        current_clear_msg = sanitizer.sanitize(
            get_latest_human_message(state.messages).content
        )
        query = optimizer_llm.invoke([
            SystemMessage(content=OPTIMIZER_PROMPT),
            HumanMessage(content=current_clear_msg),
        ]).finetuned_query

    result = retriever.invoke(query)
    return {
        "run_query": query,
        "relevant_documents": [doc.page_content for doc in result],
    }


def grade_docs(state: MainState):
    inputs = [
        {"question": state.run_query, "document": doc}
        for doc in state.relevant_documents
    ]
    grader_responses = retrieval_grader.batch(inputs) if inputs else []
    filtered_docs = [
        doc
        for doc, res in zip(state.relevant_documents, grader_responses)
        if res.binary_score.lower() == "yes"
    ]
    return {
        "relevant_documents": filtered_docs,
        "max_retrieves": state.max_retrieves + 1,
    }


def response_or_fallback(state: MainState):
    if state.relevant_documents or state.max_retrieves >= MAX_RETRIES:
        return "retrieval_response"
    return "transform_query"


def transform_query(state: MainState):
    improved_run_query = llm.invoke([
        SystemMessage(content=QUESTION_REWRITER_PROMPT),
        HumanMessage(content=f"Initial question: {state.run_query}"),
    ])
    return {"run_query": improved_run_query.content}


def retrieval_response(state: MainState):
    if not state.relevant_documents:
        return {"static_answer": "I couldn't find that in the parking information."}

    answer = retrieval_chain.invoke({
        "question": get_latest_human_message(state.messages).content,
        "context": "\n\n".join(state.relevant_documents),
    })
    return {"static_answer": answer}


# add to each ending the last check with nlp model
workflow = StateGraph(MainState, input_schema=MessagesState, output_schema=MessagesState)

for node in (
    security_stoppage,
    summarize_conversation,
    docs_retrieval,
    grade_docs,
    transform_query,
    retrieval_response,
    info_response,
    update_state,
    agent,
    tools,
    reservation_response,
):
    workflow.add_node(node)
workflow.add_node("intent_router", check_intent)

workflow.add_conditional_edges(START, route_entry, [
    "security_stoppage", "summarize_conversation", "intent_router",
])
workflow.add_edge("security_stoppage", END)
workflow.add_edge("summarize_conversation", "intent_router")
workflow.add_conditional_edges("intent_router", route_after_intent, [
    "docs_retrieval", "update_state", "agent", "info_response",
])

workflow.add_edge("docs_retrieval", "grade_docs")
workflow.add_conditional_edges("grade_docs", response_or_fallback, [
    "retrieval_response", "transform_query",
])
workflow.add_edge("transform_query", "docs_retrieval")
workflow.add_conditional_edges("retrieval_response", route_after_retrieval, [
    "update_state", "agent", "info_response",
])
workflow.add_edge("info_response", END)

workflow.add_conditional_edges("update_state", route_after_update, [
    "agent", "reservation_response",
])
workflow.add_conditional_edges("agent", route_after_agent, [
    "tools", "reservation_response", END,
])
workflow.add_edge("tools", "agent")
workflow.add_edge("reservation_response", END)

graph = workflow.compile()
