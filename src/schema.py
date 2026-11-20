from typing import Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from datetime import datetime

class ReservationPeriod(BaseModel):
    start_date: str | None = Field(description="The start date with time from which the reservation is starting", default=None)
    end_date: str | None = Field(description="The end date with time to which the reservation is due", default=None)

class UserState(BaseModel):
    place: str | None = Field(default=None, description="Requested parking venue or address")
    name: str | None = Field(description="user name", default=None)
    car_number: str | None = Field(description="user car plate", default=None)
    reservation_period: ReservationPeriod = Field(
        description="Reservation period",
        default_factory=ReservationPeriod
        )
    approval_time: datetime | None = Field(default=None)

#without approval_time for updating
class UserInfoExtraction(BaseModel):
        place: Literal['Port Łódź', 'Manufaktura', 'Galeria Łódzka', 'Nowa Sukcesja'] | None = Field(default=None, description="Name of the requested parking venue or None if not provided")
        name: str | None = None
        car_number: str | None = None
        reservation_period: ReservationPeriod = Field(
            default_factory=ReservationPeriod
        )

class MainState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    summary: str = ""
    relevant_documents: Optional[list[str]] = Field(default_factory=list)
    user_info: UserState = Field(description="user's info for the reservation", default_factory=UserState)
    missing_info: list[str] = Field(default_factory=list)
    reservation_question: str = ""
    sql_calls: int = 0
    run_query: str = ""
    max_retrieves: int = 0
    wants_static_info: bool = False
    wants_dynamic_info: bool = False
    wants_reservation: bool = False
    dynamic_question: str = ""
    static_answer: str = ""

class UpgradedQuerySchema(BaseModel):
    finetuned_query: str = Field(description='upgraded query for search')

class FraudDetectSchema(BaseModel):
    safe: Literal[True, False] = Field(description="True if the query is safe, False if it's fraudulent.")
    reason: str | None = Field(description="explanation if unsafe", default=None)

class RefinedQuerySchema(BaseModel):
    query_relevant: Literal['relevant', 'irrelevant'] = Field('relevant if query has anything to do with the parking reservation purpose, irrelevant if query is not leading to the parking reservation')

class GradeDocs(BaseModel):
    binary_score: Literal['yes', 'no'] = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class DecideRetrieval(BaseModel):
    decision: Literal[True, False] = Field(description="True if the user message requires retrieval of information.")

class IntentSchema(BaseModel):
    static_info: bool = Field(description="True for static_info If user asks about general information, parking details, location, booking process")
    dynamic_info: bool = Field(description="True if user asks about space availability, working hours, prices (the data that is frequently changes")
    reservation: bool = Field(description="True if user does a reservation or provides a new userinfo for the reservation")
    dynamic_question: str
# class IntentSchema(BaseModel):
#     intention: Literal['reservation','retrieve'] = Field(description="User's intention")
