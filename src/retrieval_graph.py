from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from pathlib import Path
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage

from src.schema import (
    RetrievalState,
    GradeDocs
    )
from src.functions import (
    dense_retriever,
    sparse_retriever,
)
from src.models_vector import get_vector, llm

data_path = Path(__file__).parent.parent / 'data'
db_path = data_path / 'parking.db'
vector_store = get_vector(data_path)
semantic_search = dense_retriever(vector_store, 4)
keyword_search = sparse_retriever(vector_store, 4)

MAX_RETRIES = 3

QUESTION_REWRITER_PROMPT = """
    You a query re-writer that converts an input from user to a better version that is optimized
    for search. Look at the input and try to reason about the underlying semantic intent / meaning.
"""
GRADE_DOCUMENTS_PROMPT = """
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

retriever = EnsembleRetriever(
    retrievers=[semantic_search, keyword_search],
    weights=[0.60, 0.40],  # 60% semantic importance, 40% keyword importance
    )

def docs_retrieval(state):
    if state.run_query:
        query = state.run_query
    else:
        query = state.quries[-1]

    result = retriever.invoke(query)
    return {'relevant_documents': [doc.page_content for doc in result]}

def grade_docs(state):
# conditional edge to check amount of retrieved docs, if 0, retry to format query and retry the retrieval and then ask for new query from user
    if state.run_query:
        query = state.run_query
    else:
        query = state.quries[-1]

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
    if len(state.relevant_documents) > 0 or state.max_retrieves > MAX_RETRIES:
        return END
    return 'transform_query'


def transform_query(state):
    improved_run_query = llm.invoke([
    SystemMessage(content=QUESTION_REWRITER_PROMPT),
    HumanMessage(content=f"Initial question: {state.queries[-1]}")
        ])
    
    return {'run_query': improved_run_query.content}

workflow = StateGraph(RetrievalState)
workflow.add_node(docs_retrieval)
workflow.add_edge(START, 'docs_retrieval')
workflow.add_node(grade_docs)
workflow.add_edge('docs_retrieval', 'grade_docs')
workflow.add_node(transform_query)
workflow.add_conditional_edges('grade_docs', response_or_fallback, [END, 'transform_query'])
retrieval_graph = workflow.compile()