# current implementation if the parallel execution for sql and 
# vector retriever is choosen and they return to the same key,
# now only toolnode for sql and state with retrieval as separate node.
from functools import lru_cache

from langchain.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "IP_ADDRESS",
]

# Public venue/street names that the English NLP model may misclassify as person names.
PII_ALLOW_LIST = [
    "Galeria Łódzka",
    "Manufaktura",
    "Port Łódź",
    "Nowa Sukcesja",
    "Łódź",
    "al",
    "al.",
    "Piłsudskiego",
    "Drewnowska",
    "Pabianicka",
    "Politechniki",
    "Arrival",
]


@lru_cache(maxsize=1)
def _get_pii_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Load the local spaCy NLP model once and return Presidio engines."""
    nlp_provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_provider.create_engine(),
        supported_languages=["en"],
    )
    return analyzer, AnonymizerEngine()


@lru_cache(maxsize=512)
def redact_pii(text: str) -> str:
    """Detect and mask PII in text using a local spaCy NLP model via Presidio."""
    if not text or not text.strip():
        return text

    analyzer, anonymizer = _get_pii_engines()
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=PII_ENTITIES,
        allow_list=PII_ALLOW_LIST,
        score_threshold=0.5,
    )
    if not results:
        return text

    return anonymizer.anonymize(text=text, analyzer_results=results).text


def redact_pii_rows(rows: list[dict]) -> list[dict]:
    """Apply redact_pii to every string value in SQL result rows."""
    return [
        {
            col: redact_pii(val) if isinstance(val, str) else val
            for col, val in row.items()
        }
        for row in rows
    ]

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