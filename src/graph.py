from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState, add_messages
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Send
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState
    )
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
from src.models_vector import llm, embedding, get_vector
from src.schema import (
    UserState,
    MainState,
    InputSchema,
    UpgradedQuerySchema,
    FraudDetectSchema,
    RefinedQuerySchema
)
from src.security import (
    InputSanitizer,
    SecutiryGuard
)
#connect to sql and chroma in runtime context 

#build nodes which
# get input - guardrails on input - llm to update its user's state if possible, fine-tune the query, 
# retrieve sql and hybrid search unsemble in parallel,
# guard rails on query in sql and quardrails on output from sql
# guard rails on retrieved documents, if passed that
# retrieve the vector_db based on query and store in contextstate class,
# now decide if missing some info in state so mention to agent what is missing and ask user + response based on context and original query.
# iterate over until the user's state if enought for reservation,
# invoke next graph with human-in-the-loop with administator to approve the reservation
# invoke next graph and based on admin response - implement reservation and answer the user or if rejected, response as well to user, maybe with admin message why.


db_path = Path(__file__).parent.parent / 'data' / 'parking.db'
sql_db = SQLDatabase.from_uri(f'sqlite:///{db_path}', include_tables=['parking_slots'])


# custom reducer, dont know will i query in parallel currently, will see
def reduce_list(left: list | None, right: list | None) -> list:
    if not left:
        left = []
    if not right:
        right = []
    return left + right

def user_input(state):
    response = llm.with_structured_output(RefinedQuerySchema).invoke([HumanMessage("check if the user query has anything to do with parking reservation")] + [state.query])
    if response.query_relevant:
        return ''


def should_summarize(state):
    pass

@tool('select_query', description=
      '''only select queries are allowed.
      "parking_slots" table with schema:
      id: Primary Key
      slot_code: TEXT
      parking_date: TEXT
      location: TEXT
      parking_name: TEXT
      spot_details: TEXT
      price_per_hour_pnl INTEGER
      opens_at: TEXT
      closes_at: TEXT
      is_available INTEGER (0 | 1)
      ''')
def query_sql(query, runtime: ToolRuntime):
    '''query sql '''
    pass


@tool
def check_state(query:str, runtime: ToolRuntime):
    '''check the user state to see for still missing fields'''
    user_info = runtime.state.user_info

@tool
def update_state(query:str, runtime: ToolRuntime):
    '''update the state'''
    pass

def should_retrieve(query: str):
    '''check user's query for uniqueness and decide either allowed to retrieve rag again '''
    pass

def hybrid_search(query:str):
    pass

workflow = AgentState(MainState, InputSchema=InputSchema)
workflow.add_node(START, )