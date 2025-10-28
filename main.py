import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from database import db, create_document, get_documents

app = FastAPI(title="AI Personal Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str
    subject: Optional[str] = None
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    user_id: str
    subject: str = Field("general")
    message: str
    history: List[ChatMessage] = []
    level: Optional[str] = Field(None, description="beginner/intermediate/advanced")


class ChatResponse(BaseModel):
    reply: str
    key_points: List[str] = []
    follow_up_question: Optional[str] = None


class QuizRequest(BaseModel):
    topic: str
    level: str = Field("beginner")
    count: int = Field(5, ge=1, le=10)


class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer_index: int
    explanation: str


class QuizResponse(BaseModel):
    topic: str
    level: str
    questions: List[QuizQuestion]


class ProgressItem(BaseModel):
    topic: str
    level: str
    score: Optional[float] = None
    notes: Optional[str] = None
    completed_at: Optional[str] = None


class SaveProgressRequest(BaseModel):
    user_id: str
    items: List[ProgressItem]


@app.get("/")
def read_root():
    return {"message": "AI Personal Tutor Backend is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, 'name', 'unknown')
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"

    return response


def _simple_tutor_reply(prompt: str, subject: str, level: Optional[str] = None, history: List[ChatMessage] = []):
    subject_prefix = subject.title() if subject else "General"
    level_line = f"For a {level} learner, " if level else ""

    # Derive a concise explanation pattern
    reply = (
        f"{subject_prefix} insight:\n"
        f"{level_line}here's a clear explanation of your question:\n\n"
        f"1) Restate: {prompt.strip()}\n"
        f"2) Core idea: Break the problem into smaller parts and tackle each with examples.\n"
        f"3) Example: If we apply this to a simple case, notice how inputs change the output step-by-step.\n\n"
        f"Tip: Try to explain it back in your own words."
    )

    key_points = [
        "Identify what's being asked",
        "Break it into smaller steps",
        "Work through a concrete example",
        "Check your result and reflect"
    ]

    follow = "What part feels unclear right now—terms, steps, or why it works?"

    return reply, key_points, follow


@app.post("/api/tutor/chat", response_model=ChatResponse)
def tutor_chat(req: ChatRequest):
    # Create assistant reply using a deterministic helper
    reply, key_points, follow = _simple_tutor_reply(req.message, req.subject, req.level, req.history)

    # Persist conversation if DB available
    try:
        if db is not None:
            now = datetime.now(timezone.utc).isoformat()
            create_document("message", {
                "user_id": req.user_id,
                "role": "user",
                "content": req.message,
                "subject": req.subject,
                "level": req.level,
                "created_at": now,
                "updated_at": now,
            })
            create_document("message", {
                "user_id": req.user_id,
                "role": "assistant",
                "content": reply,
                "subject": req.subject,
                "level": req.level,
                "created_at": now,
                "updated_at": now,
            })
    except Exception:
        # If DB is not configured, just skip persistence
        pass

    return ChatResponse(reply=reply, key_points=key_points, follow_up_question=follow)


def _generate_mcq(topic: str, level: str, idx: int) -> QuizQuestion:
    base_q = f"[{level.title()}] {topic.title()} concept check #{idx+1}"
    options = [
        f"Definition related to {topic}",
        f"Example of {topic}",
        f"Counter-example of {topic}",
        f"Irrelevant statement"
    ]
    # Deterministic correct answer rotates
    answer_index = idx % len(options)
    explanation = (
        f"Option {answer_index+1} is best because it directly addresses {topic} at a {level} level."
    )
    return QuizQuestion(question=base_q, options=options, answer_index=answer_index, explanation=explanation)


@app.post("/api/tutor/quiz", response_model=QuizResponse)
def generate_quiz(req: QuizRequest):
    questions = [_generate_mcq(req.topic, req.level, i) for i in range(req.count)]
    return QuizResponse(topic=req.topic, level=req.level, questions=questions)


@app.get("/api/progress/{user_id}")
def get_progress(user_id: str):
    try:
        if db is None:
            return {"items": []}
        docs = get_documents("progress", {"user_id": user_id}, limit=100)
        # Normalize ObjectId for JSON if present
        items = []
        for d in docs:
            d.pop("_id", None)
            items.append(d)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/progress/save")
def save_progress(req: SaveProgressRequest):
    try:
        if db is None:
            return {"saved": False, "reason": "No database configured"}
        saved_ids = []
        for item in req.items:
            doc = item.model_dump()
            doc.update({
                "user_id": req.user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            saved_id = create_document("progress", doc)
            saved_ids.append(saved_id)
        return {"saved": True, "ids": saved_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
