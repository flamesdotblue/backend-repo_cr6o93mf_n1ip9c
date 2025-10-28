import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from database import db, create_document, get_documents
from schemas import CoursePlan, Module, Lesson

# Auth & Security
from jose import JWTError, jwt
from passlib.context import CryptContext
import stripe

JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALGO = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app = FastAPI(title="AI Personal Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------
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


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    name: Optional[str] = None
    email: Optional[str] = None


class PlanRequest(BaseModel):
    user_query: str
    subject: str = "general"
    level: str = "beginner"
    target_hours: float = 6.0


# ---------- Helpers ----------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGO)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _simple_tutor_reply(prompt: str, subject: str, level: Optional[str] = None, history: List[ChatMessage] = []):
    subject_prefix = subject.title() if subject else "General"
    level_line = f"For a {level} learner, " if level else ""

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


def _generate_mcq(topic: str, level: str, idx: int) -> QuizQuestion:
    base_q = f"[{level.title()}] {topic.title()} concept check #{idx+1}"
    options = [
        f"Definition related to {topic}",
        f"Example of {topic}",
        f"Counter-example of {topic}",
        f"Irrelevant statement"
    ]
    answer_index = idx % len(options)
    explanation = (
        f"Option {answer_index+1} is best because it directly addresses {topic} at a {level} level."
    )
    return QuizQuestion(question=base_q, options=options, answer_index=answer_index, explanation=explanation)


def _generate_course_plan(req: PlanRequest) -> CoursePlan:
    # Deterministic curriculum generator with curated links
    subject = req.subject.lower()
    level = req.level.lower()

    def lesson(title: str, ltype: str, url: Optional[str], hours: float) -> Lesson:
        return Lesson(title=title, type=ltype, url=url, duration_hours=hours)

    curated = {
        "math": [
            ("Khan Academy: {topic}", "video", "https://www.khanacademy.org/", 0.5),
            ("Brilliant: Interactive {topic}", "article", "https://brilliant.org/", 0.5),
            ("MIT OCW Notes", "pdf", "https://ocw.mit.edu/", 0.5)
        ],
        "programming": [
            ("freeCodeCamp: {topic}", "video", "https://www.youtube.com/c/Freecodecamp", 0.75),
            ("MDN Web Docs", "article", "https://developer.mozilla.org/", 0.5),
            ("RealPython Guide", "article", "https://realpython.com/", 0.5)
        ],
        "science": [
            ("Crash Course: {topic}", "video", "https://www.youtube.com/user/crashcourse", 0.5),
            ("Nature Education: Primer", "article", "https://www.nature.com/scitable/", 0.5),
            ("OpenStax Textbook", "pdf", "https://openstax.org/", 0.75)
        ]
    }

    topic = req.user_query.strip().split(" ")[:3]
    topic = " ".join(topic) or subject

    curated_list = curated.get(subject, [
        ("Wikipedia Overview: {topic}", "article", "https://en.wikipedia.org/wiki/", 0.5),
        ("YouTube Lecture: {topic}", "video", "https://www.youtube.com/results?search_query=", 0.75),
        ("Open Textbook Library", "pdf", "https://open.umn.edu/opentextbooks", 0.5)
    ])

    # Build modules: Foundations, Practice, Projects
    modules: List[Module] = []

    def build_module(title: str, factor: float) -> Module:
        lessons: List[Lesson] = []
        base_hours = 0.0
        for name, ltype, base_url, h in curated_list:
            item_title = name.format(topic=topic.title())
            url = base_url
            if base_url.endswith("?") or base_url.endswith("="):
                url = f"{base_url}{topic}"
            lessons.append(lesson(item_title, ltype, url, round(h * factor, 2)))
            base_hours += h * factor
        est = round(base_hours, 2)
        return Module(title=title, lessons=lessons, estimated_hours=est)

    modules.append(build_module("Foundations", 1.0))
    modules.append(build_module("Guided Practice", 1.0 if level == "beginner" else 0.9))
    modules.append(build_module("Project & Review", 0.8))

    total_hours = round(sum(m.estimated_hours for m in modules), 2)

    # Normalize to target hours by scaling if necessary
    if total_hours > 0 and req.target_hours:
        scale = req.target_hours / total_hours
        for m in modules:
            m.estimated_hours = round(m.estimated_hours * scale, 2)
            for les in m.lessons:
                les.duration_hours = round(les.duration_hours * scale, 2)
        total_hours = round(req.target_hours, 2)

    return CoursePlan(subject=subject, level=level, total_hours=total_hours, modules=modules)


# ---------- Routes ----------
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


# Auth endpoints
@app.post("/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    users = db["user"]
    if users.find_one({"email": req.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = hash_password(req.password)
    users.insert_one({
        "name": req.name,
        "email": req.email,
        "hashed_password": hashed,
        "role": "user",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })
    token = create_access_token({"sub": req.email})
    return TokenResponse(access_token=token, name=req.name, email=req.email)


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    user = db["user"].find_one({"email": req.email})
    if not user or not verify_password(req.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": req.email})
    return TokenResponse(access_token=token, name=user.get("name"), email=user.get("email"))


@app.get("/auth/me")
def me(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        email = payload.get("sub")
        return {"email": email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Tutor chat
@app.post("/api/tutor/chat", response_model=ChatResponse)
def tutor_chat(req: ChatRequest):
    reply, key_points, follow = _simple_tutor_reply(req.message, req.subject, req.level, req.history)

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
        pass

    return ChatResponse(reply=reply, key_points=key_points, follow_up_question=follow)


# Quiz
@app.post("/api/tutor/quiz", response_model=QuizResponse)
def generate_quiz(req: QuizRequest):
    questions = [_generate_mcq(req.topic, req.level, i) for i in range(req.count)]
    return QuizResponse(topic=req.topic, level=req.level, questions=questions)


# Progress
@app.get("/api/progress/{user_id}")
def get_progress(user_id: str):
    try:
        if db is None:
            return {"items": []}
        docs = get_documents("progress", {"user_id": user_id}, limit=100)
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


# Course plan
@app.post("/api/tutor/plan", response_model=CoursePlan)
def generate_plan(req: PlanRequest):
    plan = _generate_course_plan(req)
    # Persist the plan header for progress tracking (optional)
    try:
        if db is not None:
            create_document("courseplan", {
                "subject": plan.subject,
                "level": plan.level,
                "total_hours": plan.total_hours,
                "modules_count": len(plan.modules)
            })
    except Exception:
        pass
    return plan


# Payments
class CheckoutRequest(BaseModel):
    plan: str = Field("pro")


@app.post("/api/payments/create-checkout-session")
def create_checkout_session(req: CheckoutRequest):
    if not stripe.api_key:
        raise HTTPException(status_code=400, detail="Stripe is not configured")

    try:
        if STRIPE_PRICE_ID:
            session = stripe.checkout.Session.create(
                mode="subscription",
                line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
                success_url=f"{FRONTEND_URL}/?success=true",
                cancel_url=f"{FRONTEND_URL}/?canceled=true",
            )
        else:
            session = stripe.checkout.Session.create(
                mode="payment",
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "Tutor Pro"},
                        "unit_amount": 999,
                    },
                    "quantity": 1,
                }],
                success_url=f"{FRONTEND_URL}/?success=true",
                cancel_url=f"{FRONTEND_URL}/?canceled=true",
            )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
