from pathlib import Path

from langchain.messages import HumanMessage, SystemMessage
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from src.functions import dense_retriever, redact_pii, sparse_retriever
from src.models_vector import get_vector, llm
from src.prompts import (
    GRADE_DOCUMENTS_PROMPT,
    OPTIMIZER_PROMPT,
    QUESTION_REWRITER_PROMPT,
)
from src.schema import (
    GradeDocs,
    RetrievalInput,
    RetrievalOutput,
    RetrievalState,
    UpgradedQuerySchema,
)

MAX_RETRIES = 3

DATA_PATH = Path(__file__).parent.parent / "data"
vector_store = get_vector(DATA_PATH)
retriever = EnsembleRetriever(
    retrievers=[dense_retriever(vector_store, 4), sparse_retriever(vector_store, 4)],
    weights=[0.60, 0.40],
)

optimizer_llm = llm.with_structured_output(UpgradedQuerySchema)
retrieval_grader = (
    ChatPromptTemplate.from_messages([
        ("system", GRADE_DOCUMENTS_PROMPT),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])
    | llm.with_structured_output(GradeDocs)
)


def optimize_query(state: RetrievalState):
    response = optimizer_llm.invoke([
        SystemMessage(content=OPTIMIZER_PROMPT),
        HumanMessage(content=state.current_message),
    ])
    return {"run_query": response.finetuned_query}


def docs_retrieval(state: RetrievalState):
    result = retriever.invoke(state.run_query)
    return {"relevant_documents": [redact_pii(doc.page_content) for doc in result]}


def grade_docs(state: RetrievalState):
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


def route_after_grade(state: RetrievalState):
    if state.relevant_documents or state.max_retrieves >= MAX_RETRIES:
        return END
    return "transform_query"


def transform_query(state: RetrievalState):
    improved = llm.invoke([
        SystemMessage(content=QUESTION_REWRITER_PROMPT),
        HumanMessage(content=f"Initial question: {state.run_query}"),
    ])
    return {"run_query": improved.content}


builder = StateGraph(
    RetrievalState,
    input_schema=RetrievalInput,
    output_schema=RetrievalOutput,
)
builder.add_node("optimize_query", optimize_query)
builder.add_node("docs_retrieval", docs_retrieval)
builder.add_node("grade_docs", grade_docs)
builder.add_node("transform_query", transform_query)

builder.add_edge(START, "optimize_query")
builder.add_edge("optimize_query", "docs_retrieval")
builder.add_edge("docs_retrieval", "grade_docs")
builder.add_conditional_edges("grade_docs", route_after_grade, [END, "transform_query"])
builder.add_edge("transform_query", "docs_retrieval")

retrieval_graph = builder.compile()
