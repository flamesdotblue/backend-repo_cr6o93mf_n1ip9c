from pydantic import BaseModel, Field
from typing import Optional, List

# Define collections for the tutor app

class TutorUser(BaseModel):
    name: str
    email: Optional[str] = None
    level: Optional[str] = Field(None, description="beginner/intermediate/advanced")
    subjects: Optional[List[str]] = []


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
