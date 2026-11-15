from operator import add
from typing import TypedDict, Optional, Any, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, RemoveMessage
from langchain_core.documents import Document



class UserState(BaseModel):
    name: Optional[str] = Field(description="user name", default=None)
    car_number: Optional[str] = Field(description="user car plate", default=None)
    reservation_period: Optional[str] = Field(description="start_date and end_date", default=None)
    approval_time: Optional[str] = Field(default=None)

class MainState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    query: str
    sql_queries: Optional[int] = Field(default=0)
    relevant_documents: Optional[list[Document]] = Field(default_factory=list)
    user_info: UserState = Field(description="user's info for the reservation")

class InputSchema(BaseModel):
    query: str

class UpgradedQuerySchema(BaseModel):
    finetuned_query: str = Field(description='upgraded query for search')

class FraudDetectSchema(BaseModel):
    safe: Literal[True, False] = Field(description="True if the transaction is safe, False if it's fraudulent.")
    reason: str | None = Field(description="explanation if unsafe", default=None)

class RefinedQuerySchema(BaseModel):
    query_relevant: Literal['relevant', 'irrelevant'] = Field('relevant if query has anything to do with the parking reservation purpose, irrelevant if query is not leading to the parking reservation')
