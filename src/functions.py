# current implementation if the parallel execution for sql and 
# vector retriever is choosen and they return to the same key,
# now only toolnode for sql and state with retrieval as separate node.
from langchain.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

def reduce_list(left: list | None, right: list | None) -> list:
    if not left:
        left = []
    if not right:
        right = []
    return left + right

def get_latest_human_message(messages):
    """Traverse backwards to find the most recent user message."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    # Fallback if no human message exists
    return messages[-1] if messages else None

def dense_retriever(vector_store, k):
    """Retriever for sematic-search"""
    # get dense retriever for semantic search
    return vector_store.as_retriever(search_kwargs={"k": k})

def sparse_retriever(vector_store, k):
    '''BM25 retriever for key-work search'''
    raw_data = vector_store.get()

    all_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw_data["documents"], raw_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 4
    return bm25_retriever