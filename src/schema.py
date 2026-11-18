from typing import Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from src.functions import reduce_list
from datetime import datetime

class ReservationPeriod(BaseModel):
    start_date: str | None = Field(description="The start date with time from which the reservation is starting", default=None)
    end_date: str | None = Field(description="The end date with time to which the reservation is due", default=None)

class UserState(BaseModel):
    name: str | None = Field(description="user name", default=None)
    car_number: str | None = Field(description="user car plate", default=None)
    reservation_period: ReservationPeriod = Field(
        description="Reservation period",
        default_factory=ReservationPeriod
        )
    approval_time: datetime | None = Field(default=None)

#without approval_time for updating
class UserInfoExtraction(BaseModel):
        name: str | None = None
        car_number: str | None = None
        reservation_period: ReservationPeriod = Field(
            default_factory=ReservationPeriod
        )

class MainState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    summary: str = ""
    queries: Annotated[list[str], reduce_list] = Field(default_factory=list)
    current_message: str = ""
    relevant_documents: Optional[list[Document]] = Field(default_factory=list)
    user_info: UserState = Field(description="user's info for the reservation", default_factory=UserState)
    missing_info: list[str] = Field(default_factory=list)

class InputSchema(BaseModel):
    query: str

class UpgradedQuerySchema(BaseModel):
    finetuned_query: str = Field(description='upgraded query for search')

class FraudDetectSchema(BaseModel):
    safe: Literal[True, False] = Field(description="True if the query is safe, False if it's fraudulent.")
    reason: str | None = Field(description="explanation if unsafe", default=None)

class RefinedQuerySchema(BaseModel):
    query_relevant: Literal['relevant', 'irrelevant'] = Field('relevant if query has anything to do with the parking reservation purpose, irrelevant if query is not leading to the parking reservation')

class GradeDocs(BaseModel):
    binary_score: Literal['yes', 'no'] = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class ReservationPeriod(BaseModel):
    start_date: str = Field(description="The start date with time from which the reservation is starting", default=None)
    end_date: str = Field(description="The end date with time to which the reservation is due", default=None)

class DecideRetrieval(BaseModel):
    decision: Literal[True, False] = Field(description="True if the user message requires retrieval of information.")

class SQLAgentState(BaseModel):
    sql_calls: int
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

class RetrievalState(BaseModel):
    queries: Annotated[list[str], reduce_list] = Field(default_factory=list)
    max_retrieves: int
    relevant_documents: Optional[list[Document]] = Field(default_factory=list)
    run_query: str = ""