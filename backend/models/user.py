from typing import List, Optional
from pydantic import BaseModel


class BasicInfo(BaseModel):
    name: str
    location: str
    profession: str
    age: int


class Expertise(BaseModel):
    areas: List[str]
    level: str


class Preferences(BaseModel):
    collaboration_style: str
    communication_preferences: List[str]
    work_environment: str


class User(BaseModel):
    id: str
    basic_info: BasicInfo
    interests: List[str]
    values: List[str]
    expertise: Expertise
    preferences: Preferences
    activities: List[str]


class UserQuery(BaseModel):
    query: str
    limit: Optional[int] = 5


class UserMatch(BaseModel):
    user: User
    compatibility_score: float
    match_reasons: List[str]


class UserMatchResponse(BaseModel):
    matches: List[UserMatch]
    query_understanding: str
