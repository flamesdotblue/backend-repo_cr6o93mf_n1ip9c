from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List

# Define collections for the tutor app

class TutorUser(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    level: Optional[str] = Field(None, description="beginner/intermediate/advanced")
    subjects: Optional[List[str]] = []


class User(BaseModel):
    name: str
    email: EmailStr
    hashed_password: str
    role: str = Field("user")


class Message(BaseModel):
    user_id: str
    role: str = Field(..., description="user or assistant")
    content: str
    subject: Optional[str] = None
    level: Optional[str] = None


class Progress(BaseModel):
    user_id: str
    topic: str
    level: str
    score: Optional[float] = None
    notes: Optional[str] = None
    completed_at: Optional[str] = None


class Lesson(BaseModel):
    title: str
    type: str = Field(..., description="video|article|pdf")
    url: Optional[str] = None
    duration_hours: float = 0.5


class Module(BaseModel):
    title: str
    lessons: List[Lesson]
    estimated_hours: float


class CoursePlan(BaseModel):
    subject: str
    level: str
    total_hours: float
    modules: List[Module]
