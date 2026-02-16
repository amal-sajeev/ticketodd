# AI-Powered Grievance Redressal System
# FastAPI + MongoDB + Qdrant + OpenAI

import os
import re
import uuid
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from enum import Enum
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import openai as openai_mod
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient, ReturnDocument
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PointIdsList
from openai import AsyncOpenAI
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
# Try multiple .env locations: next to this file, one level up, then cwd
_script_dir = Path(__file__).resolve().parent
_env_candidates = [
    _script_dir / ".env",            # demo/.env
    _script_dir.parent / ".env",     # ticketer-main/.env
    Path.cwd() / ".env",             # current working directory
]
_env_loaded = False
for _env_path in _env_candidates:
    if _env_path.is_file():
        load_dotenv(_env_path, override=True)
        _env_loaded = True
        break
if not _env_loaded:
    load_dotenv(override=True)  # fall back to python-dotenv's own search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    raise RuntimeError(
        "FATAL: JWT_SECRET must be set in the environment and be at least 32 characters. "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 2
EMBEDDING_DIM = 3072  # text-embedding-3-large

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Department(str, Enum):
    PANCHAYATI_RAJ = "panchayati_raj"
    RURAL_WATER_SUPPLY = "rural_water_supply"
    MGNREGS = "mgnregs"
    RURAL_HOUSING = "rural_housing"
    RURAL_LIVELIHOODS = "rural_livelihoods"
    SANITATION = "sanitation"
    INFRASTRUCTURE = "infrastructure"
    GENERAL = "general"

class OfficerCategory(str, Enum):
    BLOCK_DEV_OFFICER = "block_dev_officer"
    DISTRICT_PANCHAYAT_OFFICER = "district_panchayat_officer"
    EXECUTIVE_ENGINEER_RWSS = "executive_engineer_rwss"
    DRDA_PROJECT_DIRECTOR = "drda_project_director"
    GP_SECRETARY = "gp_secretary"
    MGNREGS_PROGRAMME_OFFICER = "mgnregs_programme_officer"
    GENERAL_OFFICER = "general_officer"
    SENIOR_OFFICER = "senior_officer"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class GrievanceStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"

class ResolutionType(str, Enum):
    AI = "ai"
    MANUAL = "manual"
    HYBRID = "hybrid"

class ResolutionTier(str, Enum):
    SELF_RESOLVABLE = "self_resolvable"
    OFFICER_ACTION = "officer_action"
    ESCALATION = "escalation"

class NoteType(str, Enum):
    INTERNAL = "internal"
    CITIZEN_FACING = "citizen_facing"

class Language(str, Enum):
    ODIA = "odia"
    HINDI = "hindi"
    ENGLISH = "english"

class UserRole(str, Enum):
    CITIZEN = "citizen"
    OFFICER = "officer"
    ADMIN = "admin"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class Geolocation(BaseModel):
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    type: Optional[str] = None
    coordinates: Optional[List[float]] = None
    @field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("Coordinates must be [longitude, latitude]")
        return v
    @model_validator(mode='before')
    @classmethod
    def check_location_format(cls, values):
        if isinstance(values, dict):
            if values.get('coordinates') is not None:
                if len(values['coordinates']) != 2:
                    raise ValueError("Coordinates must be [longitude, latitude]")
                values['longitude'] = values['coordinates'][0]
                values['latitude'] = values['coordinates'][1]
            elif values.get('latitude') is not None and values.get('longitude') is not None:
                values['coordinates'] = [values['longitude'], values['latitude']]
                values['type'] = 'Point'
        return values

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=72)
    full_name: str = Field(..., max_length=200)
    email: str = Field(..., max_length=320)
    phone: Optional[str] = Field(None, max_length=20)
    role: UserRole = UserRole.CITIZEN
    department: Optional[Department] = None

class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=200)
    email: Optional[str] = Field(None, max_length=320)
    phone: Optional[str] = Field(None, max_length=20)
    role: Optional[UserRole] = None
    department: Optional[Department] = None
    password: Optional[str] = Field(None, min_length=6, max_length=72)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    full_name: str
    email: str
    phone: Optional[str] = None
    role: UserRole
    department: Optional[str] = None
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class GrievanceCreate(BaseModel):
    title: str = Field(..., max_length=200)
    description: str = Field(..., max_length=5000)
    citizen_name: Optional[str] = None
    citizen_email: Optional[str] = None
    citizen_phone: Optional[str] = None
    is_anonymous: bool = False
    language: Language = Language.ENGLISH
    district: Optional[str] = None
    location: Optional[Geolocation] = None

class GrievanceResponse(BaseModel):
    id: str
    tracking_number: str
    title: str
    description: str
    citizen_name: Optional[str] = None
    citizen_email: Optional[str] = None
    citizen_phone: Optional[str] = None
    is_anonymous: bool = False
    language: Language
    district: Optional[str] = None
    department: Department
    priority: Priority
    officer_category: OfficerCategory
    status: GrievanceStatus
    sentiment: Sentiment
    created_at: datetime
    updated_at: datetime
    sla_deadline: Optional[datetime] = None
    ai_resolution: Optional[str] = None
    manual_resolution: Optional[str] = None
    resolution_type: Optional[ResolutionType] = None
    confidence_score: Optional[float] = None
    resolution_tier: Optional[ResolutionTier] = None
    assigned_officer: Optional[str] = None
    resolution_feedback: Optional[int] = Field(None, ge=1, le=5)
    notes: List[Dict] = Field(default_factory=list)
    location: Optional[Geolocation] = None
    citizen_user_id: Optional[str] = None

class GrievanceTrackResponse(BaseModel):
    id: str
    tracking_number: str
    title: str
    status: GrievanceStatus
    department: Department
    priority: Priority
    created_at: datetime
    updated_at: datetime
    sla_deadline: Optional[datetime] = None
    ai_resolution: Optional[str] = None
    manual_resolution: Optional[str] = None
    resolution_type: Optional[ResolutionType] = None
    resolution_tier: Optional[ResolutionTier] = None

class ChatMessage(BaseModel):
    message: str = Field(..., max_length=2000)
    language: Language = Language.ENGLISH
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    reply: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    filed_grievance: Optional[Dict[str, Any]] = None

class KnowledgeEntry(BaseModel):
    title: str = Field(..., max_length=500)
    content: str = Field(..., max_length=20000)
    category: Department

class ServiceMemoryEntry(BaseModel):
    query: str = Field(..., max_length=5000)
    resolution: str = Field(..., max_length=10000)
    category: Department
    agent_name: str = Field(..., max_length=200)

class SchemeEntry(BaseModel):
    name: str = Field(..., max_length=500)
    description: str = Field(..., max_length=10000)
    eligibility: str = Field(..., max_length=5000)
    department: Department
    how_to_apply: str = Field(..., max_length=5000)
    eligibility_questions: List[Dict[str, str]] = Field(default_factory=list)

class SchemeResponse(BaseModel):
    id: str; name: str; description: str; eligibility: str
    department: str; how_to_apply: str; created_at: str
    eligibility_questions: List[Dict[str, str]] = Field(default_factory=list)

class GenerateQuestionsRequest(BaseModel):
    eligibility: str
    scheme_name: str = ""

class UpdateQuestionsRequest(BaseModel):
    questions: List[Dict[str, str]]

class DocumentationResponse(BaseModel):
    id: str; title: str; content: str; category: str; created_at: str

class ServiceMemoryResponse(BaseModel):
    id: str; query: str; resolution: str; category: str; agent_name: str; created_at: str
    tracking_number: Optional[str] = None
    grievance_id: Optional[str] = None

class ManualResolution(BaseModel):
    resolution: str = Field(..., max_length=10000)
    officer: str = Field(..., max_length=200)
    add_to_service_memory: bool = True

class ResolutionFeedback(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=2000)

class GrievanceNote(BaseModel):
    content: str = Field(..., max_length=5000)
    officer: str = Field(..., max_length=200)
    note_type: NoteType = NoteType.INTERNAL

class GrievanceAssignment(BaseModel):
    officer_name: str = Field(..., max_length=200)

class AnalyticsResponse(BaseModel):
    total_grievances: int
    self_resolved: int = 0     # tier 1 confirmed by citizen
    ai_drafted: int = 0        # grievances with AI draft available
    escalated_to_human: int
    avg_resolution_time: float
    pending_count: int = 0
    sla_breached_count: int = 0
    status_distribution: Dict[str, int] = Field(default_factory=dict)
    sentiment_distribution: Dict[str, int]
    department_distribution: Dict[str, int]
    top_districts: List[Dict[str, Any]]
    top_complaints: List[Dict[str, Any]]

class GeoAnalyticsResponse(BaseModel):
    hotspot_coordinates: List[Dict[str, float]]
    region_distribution: Dict[str, int]
    district_distribution: Dict[str, int]

# ---------------------------------------------------------------------------
# App & Globals
# ---------------------------------------------------------------------------
app = FastAPI(title="Panchayati Raj & Drinking Water Department — Grievance Portal")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(self), camera=(), microphone=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "img-src 'self' data:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' https://cdn.jsdelivr.net; "
            "frame-ancestors 'none'"
        )
        return response

app.add_middleware(SecurityHeadersMiddleware)
db_client = None
db = None
qdrant = None
executor = ThreadPoolExecutor(max_workers=10)

from jinja2 import Environment, FileSystemLoader, select_autoescape
_jinja_env = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=select_autoescape(["html", "htm", "xml"]),
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env = _jinja_env
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_db()
    await startup_qdrant()
    logger.info("OpenAI model: %s | Embedding: %s", OPENAI_MODEL, EMBEDDING_MODEL)
    yield
    if db_client:
        db_client.close()

app.router.lifespan_context = lifespan

async def startup_db():
    global db_client, db
    db_client = MongoClient(MONGODB_URL)
    db = db_client.grievance_system
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, db.grievances.create_index, "created_at")
    await loop.run_in_executor(executor, db.grievances.create_index, "status")
    await loop.run_in_executor(executor, db.grievances.create_index, "department")
    await loop.run_in_executor(executor, db.grievances.create_index, "priority")
    await loop.run_in_executor(executor, db.grievances.create_index, "tracking_number")
    await loop.run_in_executor(executor, lambda: db.users.create_index([("username", 1)], unique=True))
    logger.info("Database initialized")

async def startup_qdrant():
    global qdrant
    if QDRANT_API_KEY:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    else:
        qdrant = QdrantClient(url=QDRANT_URL)
    for collection in ["documentation", "service_memory", "schemes"]:
        try:
            existing = qdrant.get_collections()
            names = [c.name for c in existing.collections]
            if collection not in names:
                qdrant.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
                logger.info("Created collection: %s", collection)
        except Exception as e:
            logger.error("Qdrant collection %s error: %s", collection, e)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
async def get_db():
    return db

async def get_qdrant():
    return qdrant

# ---------------------------------------------------------------------------
# Auth Helpers
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password cannot exceed 72 bytes")
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

_token_blacklist: set = set()

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db=Depends(get_db)):
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if token in _token_blacklist:
        raise HTTPException(status_code=401, detail="Token has been revoked")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    loop = asyncio.get_event_loop()
    user = await loop.run_in_executor(executor, db.users.find_one, {"username": username})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme), db=Depends(get_db)):
    if token is None:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, db.users.find_one, {"username": username})
    except JWTError:
        return None

def require_role(*roles):
    async def role_checker(user=Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker

def user_to_response(user: dict) -> UserResponse:
    return UserResponse(
        id=str(user["_id"]), username=user["username"], full_name=user["full_name"],
        email=user["email"], phone=user.get("phone"), role=user["role"],
        department=user.get("department"), created_at=user["created_at"])

# ---------------------------------------------------------------------------
# AI Helpers: Cache, Retry, Truncation
# ---------------------------------------------------------------------------
_embedding_cache: Dict[str, List[float]] = {}

def truncate_text(text: str, max_chars: int = 3000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

async def get_openai_embedding(text: str) -> List[float]:
    text = truncate_text(text, 2000)
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _embedding_cache:
        return _embedding_cache[key]
    response = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vec = response.data[0].embedding
    _embedding_cache[key] = vec
    if len(_embedding_cache) > 500:
        oldest = next(iter(_embedding_cache))
        del _embedding_cache[oldest]
    return vec

async def openai_chat(messages: list, json_mode: bool = False, max_retries: int = 3,
                      **extra_kwargs) -> Optional[str]:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(max_retries):
        try:
            resp = await openai_client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, **kwargs)
            return resp.choices[0].message.content.strip()
        except (openai_mod.RateLimitError, openai_mod.APIConnectionError) as e:
            logger.warning("OpenAI retry %d: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logger.error("OpenAI error: %s", e)
            return None
    return None

# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------
def generate_tracking_number(db) -> str:
    import secrets
    import string
    counter = db.counters.find_one_and_update(
        {"_id": "grievance"}, {"$inc": {"seq": 1}},
        upsert=True, return_document=ReturnDocument.AFTER)
    # Append a random alphanumeric suffix to prevent enumeration
    random_suffix = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    return f"GRV-{datetime.now(timezone.utc).year}-{counter['seq']:06d}{random_suffix}"

def calculate_sla_deadline(priority: Priority) -> datetime:
    hours_map = {Priority.URGENT: 24, Priority.HIGH: 72, Priority.MEDIUM: 168, Priority.LOW: 360}
    return datetime.now(timezone.utc) + timedelta(hours=hours_map.get(priority, 168))

def convert_db_grievance(g: dict) -> dict:
    if g.get("location"):
        g["location"] = {
            "type": g["location"].get("type"),
            "coordinates": g["location"].get("coordinates"),
            "longitude": g["location"]["coordinates"][0] if g["location"].get("coordinates") else None,
            "latitude": g["location"]["coordinates"][1] if g["location"].get("coordinates") else None,
        }
    return g

LANGUAGE_NAMES = {"odia": "Odia (ଓଡ଼ିଆ)", "hindi": "Hindi (हिन्दी)", "english": "English"}

# ---------------------------------------------------------------------------
# Input Sanitization Helpers
# ---------------------------------------------------------------------------
ODISHA_DISTRICTS = [
    "Angul","Balangir","Balasore","Bargarh","Bhadrak","Boudh","Cuttack","Deogarh",
    "Dhenkanal","Gajapati","Ganjam","Jagatsinghpur","Jajpur","Jharsuguda","Kalahandi",
    "Kandhamal","Kendrapara","Kendujhar","Khordha","Koraput","Malkangiri","Mayurbhanj",
    "Nabarangpur","Nayagarh","Nuapada","Puri","Rayagada","Sambalpur","Subarnapur","Sundargarh"
]

def sanitize_str(value: str) -> str:
    """Ensure a value is a plain string, not a dict/list that could be a NoSQL operator."""
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="Invalid parameter type")
    return str(value)

def validate_uuid(value: str, param_name: str = "id") -> str:
    """Validate that a string is a valid UUID format."""
    value = sanitize_str(value)
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid {param_name} format")
    return value

# ---------------------------------------------------------------------------
# AI Service Classes
# ---------------------------------------------------------------------------
class SentimentAnalyzer:
    @staticmethod
    async def analyze_sentiment(text: str) -> Sentiment:
        text = truncate_text(text)
        prompt = (
            "Analyze the sentiment of this citizen grievance about government services. "
            "Classify as exactly one of: positive, neutral, negative, frustrated.\n\n"
            f'Grievance: "{text}"\n\nRespond with only one word.'
        )
        try:
            result = await openai_chat([{"role": "user", "content": prompt}])
            if result:
                r = result.lower()
                if "frustrated" in r: return Sentiment.FRUSTRATED
                if "negative" in r: return Sentiment.NEGATIVE
                if "positive" in r: return Sentiment.POSITIVE
            return Sentiment.NEUTRAL
        except Exception as e:
            logger.error("Sentiment analysis error: %s", e)
            return Sentiment.NEUTRAL

class GrievanceClassifier:
    @staticmethod
    async def classify(title: str, description: str) -> tuple:
        title = truncate_text(title, 200)
        description = truncate_text(description, 2800)
        prompt = (
            "Classify this citizen grievance for the Odisha Panchayati Raj & Drinking Water Department Grievance Portal.\n\n"
            f'Title: "{title}"\nDescription: "{description}"\n\n'
            "Return a JSON object with exactly these keys:\n"
            '- "department": one of [panchayati_raj, rural_water_supply, mgnregs, rural_housing, rural_livelihoods, sanitation, infrastructure, general]\n'
            '- "priority": one of [low, medium, high, urgent]\n'
            '- "officer": one of [block_dev_officer, district_panchayat_officer, executive_engineer_rwss, drda_project_director, gp_secretary, mgnregs_programme_officer, general_officer, senior_officer]\n'
            '- "tier": one of [self_resolvable, officer_action, escalation]\n\n'
            "Department guide: panchayati_raj=GP/Block/Zilla Parishad governance/elections/meetings/Sarpanch issues, "
            "rural_water_supply=Jal Jeevan Mission/Basudha/RWSS/bore wells/pipelines/water quality, "
            "mgnregs=MGNREGA job cards/wage payments/worksite complaints, "
            "rural_housing=PMAY-Gramin/Biju Pucca Ghar/Nirman Shramik housing, "
            "rural_livelihoods=NRLM/OLM/SHG loans/Mission Shakti/skill development, "
            "sanitation=Swachh Bharat Mission Gramin/ODF/toilet construction/SLWM, "
            "infrastructure=BGBO scheme/rural roads/bridges/community buildings/Finance Commission grants, "
            "general=other.\n"
            "Priority guide: urgent=life-threatening/water contamination, high=significant hardship, medium=standard, low=info request.\n"
            "Tier guide: self_resolvable=citizen can fix it themselves with correct info or guidance "
            "(application status queries, document requirements, portal/login issues, process questions, "
            "eligibility queries), officer_action=needs physical inspection, repair, dispatch, approval, "
            "or any government action, escalation=sensitive/corruption/fund misuse/repeated complaints/"
            "life-threatening situations."
        )
        try:
            result = await openai_chat([{"role": "user", "content": prompt}], json_mode=True)
            if result:
                data = json.loads(result)
                dept = Department(data["department"]) if data.get("department") in [e.value for e in Department] else Department.GENERAL
                pri = Priority(data["priority"]) if data.get("priority") in [e.value for e in Priority] else Priority.MEDIUM
                off = OfficerCategory(data["officer"]) if data.get("officer") in [e.value for e in OfficerCategory] else OfficerCategory.GENERAL_OFFICER
                tier = ResolutionTier(data["tier"]) if data.get("tier") in [e.value for e in ResolutionTier] else ResolutionTier.OFFICER_ACTION
                return dept, pri, off, tier
        except Exception as e:
            logger.error("Classification error: %s", e)
        return Department.GENERAL, Priority.MEDIUM, OfficerCategory.GENERAL_OFFICER, ResolutionTier.OFFICER_ACTION

class KnowledgeSearcher:
    @staticmethod
    async def search_documentation(query: str, limit: int = 5) -> List[Dict]:
        try:
            query_vector = await get_openai_embedding(query)
            if not qdrant: return []
            results = qdrant.query_points(collection_name="documentation", query=query_vector,
                                    limit=limit, score_threshold=0.4, with_payload=True, with_vectors=False)
            return [{"content": p.payload.get("content", ""), "title": p.payload.get("title", ""),
                      "category": p.payload.get("category", ""), "score": p.score} for p in results.points]
        except Exception as e:
            logger.error("Documentation search error: %s", e)
            return []

    @staticmethod
    async def search_service_memory(query: str, limit: int = 5) -> List[Dict]:
        try:
            query_vector = await get_openai_embedding(query)
            if not qdrant: return []
            results = qdrant.query_points(collection_name="service_memory", query=query_vector,
                                    limit=limit, score_threshold=0.45, with_payload=True, with_vectors=False)
            return [{"query": p.payload.get("query", ""), "resolution": p.payload.get("resolution", ""),
                      "category": p.payload.get("category", ""), "score": p.score} for p in results.points]
        except Exception as e:
            logger.error("Service memory search error: %s", e)
            return []

    @staticmethod
    async def search_schemes(query: str, limit: int = 5) -> List[Dict]:
        try:
            query_vector = await get_openai_embedding(query)
            if not qdrant: return []
            results = qdrant.query_points(collection_name="schemes", query=query_vector,
                                    limit=limit, score_threshold=0.35, with_payload=True, with_vectors=False)
            return [{"id": str(p.id), "name": p.payload.get("name", ""), "description": p.payload.get("description", ""),
                      "eligibility": p.payload.get("eligibility", ""),
                      "how_to_apply": p.payload.get("how_to_apply", ""),
                      "department": p.payload.get("department", ""),
                      "created_at": p.payload.get("created_at", ""),
                      "eligibility_questions": p.payload.get("eligibility_questions", []),
                      "score": p.score} for p in results.points]
        except Exception as e:
            logger.error("Schemes search error: %s", e)
            return []

class ResolutionGenerator:
    @staticmethod
    def _build_context(context_sources: List[Dict]) -> str:
        if not context_sources:
            return ""
        text = "\n\nRelevant knowledge:\n"
        for source in context_sources[:3]:
            if source.get("resolution"):
                text += f"- Previous resolution: {truncate_text(source['resolution'], 500)}\n"
            elif source.get("content"):
                text += f"- Documentation: {truncate_text(source['content'], 500)}\n"
            elif source.get("how_to_apply"):
                text += f"- Scheme: {source['name']} - {truncate_text(source['description'], 300)}\n"
        return text

    @staticmethod
    def _calc_confidence(context_sources: List[Dict], text: str) -> float:
        confidence = 0.3
        if context_sources:
            confidence += max(s.get("score", 0) for s in context_sources) * 0.4
        if text and len(text) > 50:
            confidence += 0.2
        return min(confidence, 0.95)

    @staticmethod
    async def generate_citizen_guidance(title: str, description: str, citizen_name: str,
                                         sentiment: str, language: str, context_sources: List[Dict]) -> tuple:
        """Generate self-help guidance addressed directly to the citizen."""
        context_text = ResolutionGenerator._build_context(context_sources)
        lang_name = LANGUAGE_NAMES.get(language, "English")
        prompt = (
            f"You are Seva, a helpful government assistant for the Odisha Panchayati Raj & Drinking Water Department.\n"
            f"Respond in {lang_name}. Use Markdown formatting.\n\n"
            "You are providing guidance DIRECTLY TO THE CITIZEN to help them resolve this issue themselves.\n\n"
            "CRITICAL RULES:\n"
            "- Give clear, actionable steps the citizen can take RIGHT NOW.\n"
            "- Be specific: mention which office to visit, which documents to bring, which website/portal to use.\n"
            "- NEVER claim any government action has been taken.\n"
            "- NEVER invent reference numbers or case IDs.\n"
            "- Use friendly but professional tone — you are helping them, not lecturing.\n\n"
            "FORMATTING RULES:\n"
            "- Keep the total response under 150 words.\n"
            "- Structure: one line acknowledging the issue, then a '### What You Can Do' section with 2-4 "
            "numbered steps, then one closing line offering further help if needed.\n"
            "- Use **bold** for key terms, office names, and documents.\n"
            "- Do not repeat the grievance description back.\n\n"
            f"**Issue:** {truncate_text(title, 200)} — {truncate_text(description, 1500)}\n"
            f"{context_text}\n"
            f"Citizen: {citizen_name or 'Citizen'} | Sentiment: {sentiment}"
        )
        try:
            text = await openai_chat([{"role": "user", "content": prompt}])
            if not text:
                return "Your grievance has been noted and will be reviewed by a government officer.", 0.0
            return text, ResolutionGenerator._calc_confidence(context_sources, text)
        except Exception as e:
            logger.error("Citizen guidance generation error: %s", e)
            return "Your grievance has been noted and will be reviewed by a government officer.", 0.0

    @staticmethod
    async def generate_officer_draft(title: str, description: str, citizen_name: str,
                                      sentiment: str, language: str, context_sources: List[Dict]) -> tuple:
        """Generate a draft response for an officer to review before sending."""
        context_text = ResolutionGenerator._build_context(context_sources)
        lang_name = LANGUAGE_NAMES.get(language, "English")
        prompt = (
            f"You are an AI assistant drafting a response for an officer in the Odisha "
            f"Panchayati Raj & Drinking Water Department. The officer will review and edit this before sending.\n"
            f"Draft in {lang_name}. Use Markdown formatting.\n\n"
            "CRITICAL RULES:\n"
            "- This is a DRAFT for officer review — not a final response.\n"
            "- Suggest what actions the department should take.\n"
            "- NEVER claim any action has already been taken.\n"
            "- NEVER invent reference numbers, case IDs, or tracking codes.\n"
            "- Use future tense: 'We will dispatch...', 'The department will investigate...'.\n"
            "- Only mention specific departments/agencies if clearly relevant.\n\n"
            "FORMATTING RULES:\n"
            "- Keep the total response under 150 words.\n"
            "- Structure: one line acknowledging receipt, then a '### Recommended Actions' section with "
            "2-4 bullet points for the officer to act on, then one closing line.\n"
            "- Use **bold** for key terms and deadlines.\n"
            "- Be direct and professional. Do not repeat the grievance description back.\n\n"
            f"**Grievance:** {truncate_text(title, 200)} — {truncate_text(description, 1500)}\n"
            f"{context_text}\n"
            f"Citizen: {citizen_name or 'Anonymous'} | Sentiment: {sentiment}"
        )
        try:
            text = await openai_chat([{"role": "user", "content": prompt}])
            if not text:
                return "Your grievance has been noted and will be reviewed by a government officer.", 0.0
            return text, ResolutionGenerator._calc_confidence(context_sources, text)
        except Exception as e:
            logger.error("Officer draft generation error: %s", e)
            return "Your grievance has been noted and will be reviewed by a government officer.", 0.0

# ---------------------------------------------------------------------------
# Eligibility Question Generator
# ---------------------------------------------------------------------------
async def generate_eligibility_questions(eligibility_text: str, scheme_name: str) -> list:
    prompt = (
        f'For the government scheme "{scheme_name}", the eligibility criteria are:\n'
        f'"{truncate_text(eligibility_text, 1500)}"\n\n'
        'Generate 3-5 simple yes/no questions that determine if a citizen is eligible. '
        'Each question should be easy for a common citizen to understand. Return JSON: '
        '{"questions": [{"question": "...", "eligible_answer": "yes"}, ...]}'
    )
    try:
        result = await openai_chat([{"role": "user", "content": prompt}], json_mode=True)
        if not result:
            return []
        data = json.loads(result)
        questions = data.get("questions", data) if isinstance(data, dict) else data
        if isinstance(questions, list):
            return [{"question": q.get("question", ""), "eligible_answer": q.get("eligible_answer", "yes")}
                    for q in questions if q.get("question")]
        return []
    except Exception as e:
        logger.error("Eligibility question generation error: %s", e)
        return []

# ---------------------------------------------------------------------------
# Core Grievance Processing
# ---------------------------------------------------------------------------
SELF_RESOLVE_CONFIDENCE_THRESHOLD = 0.90

async def process_grievance(data: GrievanceCreate, db, user: Optional[dict] = None) -> GrievanceResponse:
    text = f"{data.title} {data.description}"
    sentiment = await SentimentAnalyzer.analyze_sentiment(text)
    department, priority, officer_category, tier = await GrievanceClassifier.classify(data.title, data.description)
    if sentiment == Sentiment.FRUSTRATED and priority in [Priority.LOW, Priority.MEDIUM]:
        priority = Priority.HIGH

    # Override tier to escalation for urgent+frustrated
    if sentiment == Sentiment.FRUSTRATED and priority == Priority.URGENT:
        tier = ResolutionTier.ESCALATION
    if tier == ResolutionTier.ESCALATION:
        priority = max(priority, Priority.HIGH, key=lambda p: [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.URGENT].index(p))

    doc_results = await KnowledgeSearcher.search_documentation(text)
    memory_results = await KnowledgeSearcher.search_service_memory(text)
    scheme_results = await KnowledgeSearcher.search_schemes(text)
    all_context = memory_results + doc_results + scheme_results

    ai_resolution = None
    confidence_score = 0.0
    citizen_name = data.citizen_name if not data.is_anonymous else None

    if all_context:
        if tier == ResolutionTier.SELF_RESOLVABLE:
            ai_resolution, confidence_score = await ResolutionGenerator.generate_citizen_guidance(
                data.title, data.description, citizen_name,
                sentiment.value, data.language.value, all_context)
            # If confidence too low for auto-send, demote to officer_action
            if confidence_score < SELF_RESOLVE_CONFIDENCE_THRESHOLD:
                tier = ResolutionTier.OFFICER_ACTION
        else:
            ai_resolution, confidence_score = await ResolutionGenerator.generate_officer_draft(
                data.title, data.description, citizen_name,
                sentiment.value, data.language.value, all_context)

    # Status: never auto-resolve; escalation tier → escalated, everything else → pending
    g_status = GrievanceStatus.ESCALATED if tier == ResolutionTier.ESCALATION else GrievanceStatus.PENDING

    tracking = generate_tracking_number(db)
    sla = calculate_sla_deadline(priority)
    doc = {
        "_id": str(uuid.uuid4()), "tracking_number": tracking,
        "title": data.title, "description": data.description,
        "citizen_name": None if data.is_anonymous else data.citizen_name,
        "citizen_email": None if data.is_anonymous else data.citizen_email,
        "citizen_phone": None if data.is_anonymous else data.citizen_phone,
        "is_anonymous": data.is_anonymous, "language": data.language.value,
        "district": data.district, "department": department.value,
        "priority": priority.value, "officer_category": officer_category.value,
        "status": g_status.value, "sentiment": sentiment.value,
        "resolution_tier": tier.value,
        "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
        "sla_deadline": sla, "ai_resolution": ai_resolution,
        "manual_resolution": None, "resolution_type": None,
        "confidence_score": confidence_score, "assigned_officer": None,
        "resolution_feedback": None, "notes": [],
        "location": {"type": "Point", "coordinates": [data.location.longitude, data.location.latitude]}
            if data.location and data.location.longitude and data.location.latitude else None,
        "citizen_user_id": str(user["_id"]) if user else None,
    }
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, db.grievances.insert_one, doc)
    return GrievanceResponse(**doc, id=doc["_id"])

# ---------------------------------------------------------------------------
# AUTH ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/auth/register", response_model=TokenResponse)
@limiter.limit("3/minute")
async def register(request: Request, user_data: UserCreate, db=Depends(get_db)):
    # Public registration is citizen-only; officers/admins created via admin panel
    if user_data.role != UserRole.CITIZEN:
        raise HTTPException(status_code=403, detail="Public registration is for citizens only. Officer/admin accounts must be created by an administrator.")
    loop = asyncio.get_event_loop()
    existing = await loop.run_in_executor(executor, db.users.find_one, {"username": user_data.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user_doc = {
        "_id": str(uuid.uuid4()), "username": user_data.username,
        "hashed_password": hash_password(user_data.password),
        "full_name": user_data.full_name, "email": user_data.email,
        "phone": user_data.phone, "role": user_data.role.value,
        "department": user_data.department.value if user_data.department else None,
        "created_at": datetime.now(timezone.utc),
    }
    await loop.run_in_executor(executor, db.users.insert_one, user_doc)
    token = create_access_token({"sub": user_data.username, "role": user_data.role.value})
    return TokenResponse(access_token=token, user=user_to_response(user_doc))

@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, form: UserLogin, db=Depends(get_db)):
    loop = asyncio.get_event_loop()
    user = await loop.run_in_executor(executor, db.users.find_one, {"username": form.username})
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return TokenResponse(access_token=token, user=user_to_response(user))

@app.get("/auth/me", response_model=UserResponse)
async def get_me(user=Depends(get_current_user)):
    return user_to_response(user)

@app.post("/auth/logout")
async def logout(token: Optional[str] = Depends(oauth2_scheme)):
    if token:
        _token_blacklist.add(token)
        # Prune blacklist if it grows too large (expired tokens don't matter)
        if len(_token_blacklist) > 10000:
            _token_blacklist.clear()
    return {"detail": "Logged out successfully"}

@app.get("/auth/officers", response_model=List[UserResponse])
async def list_officers(user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value)),
                        db=Depends(get_db)):
    loop = asyncio.get_event_loop()
    officers = await loop.run_in_executor(
        executor, lambda: list(db.users.find({"role": {"$in": ["officer", "admin"]}})))
    return [user_to_response(o) for o in officers]

# ---------------------------------------------------------------------------
# ADMIN USER MANAGEMENT ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/admin/users", response_model=List[UserResponse])
async def admin_list_users(role: Optional[str] = None,
                           user=Depends(require_role(UserRole.ADMIN.value)),
                           db=Depends(get_db)):
    query = {}
    if role and role in [r.value for r in UserRole]:
        query["role"] = role
    loop = asyncio.get_event_loop()
    users = await loop.run_in_executor(executor, lambda: list(db.users.find(query).sort("created_at", -1)))
    return [user_to_response(u) for u in users]

@app.post("/admin/users", response_model=UserResponse)
async def admin_create_user(user_data: UserCreate,
                            user=Depends(require_role(UserRole.ADMIN.value)),
                            db=Depends(get_db)):
    loop = asyncio.get_event_loop()
    existing = await loop.run_in_executor(executor, db.users.find_one, {"username": user_data.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user_doc = {
        "_id": str(uuid.uuid4()), "username": user_data.username,
        "hashed_password": hash_password(user_data.password),
        "full_name": user_data.full_name, "email": user_data.email,
        "phone": user_data.phone, "role": user_data.role.value,
        "department": user_data.department.value if user_data.department else None,
        "created_at": datetime.now(timezone.utc),
    }
    await loop.run_in_executor(executor, db.users.insert_one, user_doc)
    logger.info("Admin %s created user %s (%s)", user["username"], user_data.username, user_data.role.value)
    return user_to_response(user_doc)

@app.put("/admin/users/{user_id}", response_model=UserResponse)
async def admin_update_user(user_id: str, update: UserUpdate,
                            user=Depends(require_role(UserRole.ADMIN.value)),
                            db=Depends(get_db)):
    user_id = validate_uuid(user_id, "user_id")
    loop = asyncio.get_event_loop()
    target = await loop.run_in_executor(executor, db.users.find_one, {"_id": user_id})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    set_fields: Dict[str, Any] = {}
    if update.full_name is not None:
        set_fields["full_name"] = update.full_name
    if update.email is not None:
        set_fields["email"] = update.email
    if update.phone is not None:
        set_fields["phone"] = update.phone
    if update.role is not None:
        set_fields["role"] = update.role.value
    if update.department is not None:
        set_fields["department"] = update.department.value
    if update.password is not None:
        set_fields["hashed_password"] = hash_password(update.password)
    if not set_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = await loop.run_in_executor(
        executor, lambda: db.users.update_one({"_id": user_id}, {"$set": set_fields}))
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    updated = await loop.run_in_executor(executor, db.users.find_one, {"_id": user_id})
    logger.info("Admin %s updated user %s", user["username"], target["username"])
    return user_to_response(updated)

@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: str,
                            user=Depends(require_role(UserRole.ADMIN.value)),
                            db=Depends(get_db)):
    user_id = validate_uuid(user_id, "user_id")
    if str(user["_id"]) == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    loop = asyncio.get_event_loop()
    target = await loop.run_in_executor(executor, db.users.find_one, {"_id": user_id})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    await loop.run_in_executor(executor, db.users.delete_one, {"_id": user_id})
    logger.info("Admin %s deleted user %s", user["username"], target["username"])
    return {"detail": f"User '{target['username']}' deleted"}

# ---------------------------------------------------------------------------
# GRIEVANCE ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/grievances", response_model=GrievanceResponse)
async def create_grievance(data: GrievanceCreate, background_tasks: BackgroundTasks,
                           user=Depends(get_optional_user), db=Depends(get_db)):
    try:
        return await process_grievance(data, db, user)
    except Exception as e:
        logger.error("Error creating grievance: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/grievances", response_model=List[GrievanceResponse])
async def get_grievances(
    status: Optional[GrievanceStatus] = None, department: Optional[Department] = None,
    priority: Optional[Priority] = None, district: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100), skip: int = Query(0, ge=0, le=10000),
    user=Depends(get_current_user), db=Depends(get_db)):
    try:
        fq = {}
        if status: fq["status"] = status.value
        if department: fq["department"] = department.value
        if priority: fq["priority"] = priority.value
        if district:
            district = sanitize_str(district)
            if district not in ODISHA_DISTRICTS:
                raise HTTPException(status_code=400, detail="Invalid district name")
            fq["district"] = district
        if user["role"] == UserRole.CITIZEN.value:
            fq["citizen_user_id"] = str(user["_id"])
        def fetch():
            return [convert_db_grievance(g) for g in
                    db.grievances.find(fq).sort("created_at", -1).skip(skip).limit(limit)]
        loop = asyncio.get_event_loop()
        grievances = await loop.run_in_executor(executor, fetch)
        return [GrievanceResponse(**g, id=g["_id"]) for g in grievances]
    except Exception as e:
        logger.error("Error getting grievances: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/grievances/track/{tracking_number}", response_model=GrievanceTrackResponse)
@limiter.limit("10/minute")
async def track_grievance(request: Request, tracking_number: str, db=Depends(get_db)):
    tracking_number = sanitize_str(tracking_number)
    if not re.match(r'^GRV-\d{4}-[A-Za-z0-9]{6,16}$', tracking_number):
        raise HTTPException(status_code=400, detail="Invalid tracking number format")
    loop = asyncio.get_event_loop()
    g = await loop.run_in_executor(executor, db.grievances.find_one, {"tracking_number": tracking_number})
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return GrievanceTrackResponse(
        id=g["_id"], tracking_number=g["tracking_number"], title=g["title"], status=g["status"],
        department=g["department"], priority=g["priority"], created_at=g["created_at"],
        updated_at=g["updated_at"], sla_deadline=g.get("sla_deadline"),
        ai_resolution=g.get("ai_resolution"), manual_resolution=g.get("manual_resolution"),
        resolution_type=g.get("resolution_type"),
        resolution_tier=g.get("resolution_tier"))

@app.get("/grievances/{grievance_id}", response_model=GrievanceResponse)
async def get_grievance(grievance_id: str, user=Depends(get_current_user), db=Depends(get_db)):
    grievance_id = validate_uuid(grievance_id, "grievance_id")
    loop = asyncio.get_event_loop()
    g = await loop.run_in_executor(executor, db.grievances.find_one, {"_id": grievance_id})
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    if user["role"] == UserRole.CITIZEN.value and g.get("citizen_user_id") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Access denied")
    return GrievanceResponse(**convert_db_grievance(g), id=g["_id"])

@app.put("/grievances/{grievance_id}/resolve", response_model=GrievanceResponse)
async def resolve_grievance(grievance_id: str, resolution: ManualResolution,
                             background_tasks: BackgroundTasks,
                             user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value)),
                             db=Depends(get_db)):
    grievance_id = validate_uuid(grievance_id, "grievance_id")
    update_data = {
        "manual_resolution": resolution.resolution, "resolution_type": ResolutionType.MANUAL.value,
        "status": GrievanceStatus.RESOLVED.value, "assigned_officer": resolution.officer,
        "updated_at": datetime.now(timezone.utc)}
    if resolution.add_to_service_memory:
        loop2 = asyncio.get_event_loop()
        g = await loop2.run_in_executor(executor, db.grievances.find_one, {"_id": grievance_id})
        if g:
            background_tasks.add_task(add_to_service_memory, g, resolution.resolution, resolution.officer)
    def update():
        return db.grievances.update_one({"_id": grievance_id}, {"$set": update_data})
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, update)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return await get_grievance(grievance_id, user, db)

@app.post("/grievances/{grievance_id}/notes", response_model=GrievanceResponse)
async def add_note(grievance_id: str, note: GrievanceNote,
                   user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value)),
                   db=Depends(get_db)):
    grievance_id = validate_uuid(grievance_id, "grievance_id")
    new_note = {"id": str(uuid.uuid4()), "content": note.content, "officer": note.officer,
                "note_type": note.note_type.value, "created_at": datetime.now(timezone.utc)}
    def update():
        return db.grievances.update_one({"_id": grievance_id},
            {"$push": {"notes": new_note}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, update)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return await get_grievance(grievance_id, user, db)

@app.put("/grievances/{grievance_id}/status", response_model=GrievanceResponse)
async def update_status(grievance_id: str, new_status: GrievanceStatus,
                        user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value)),
                        db=Depends(get_db)):
    grievance_id = validate_uuid(grievance_id, "grievance_id")
    def update():
        return db.grievances.update_one({"_id": grievance_id},
            {"$set": {"status": new_status.value, "updated_at": datetime.now(timezone.utc)}})
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, update)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return await get_grievance(grievance_id, user, db)

@app.post("/grievances/{grievance_id}/feedback", response_model=GrievanceResponse)
async def add_feedback(grievance_id: str, feedback: ResolutionFeedback,
                       user=Depends(get_current_user), db=Depends(get_db)):
    grievance_id = validate_uuid(grievance_id, "grievance_id")
    # Verify ownership: only the citizen who filed this grievance can leave feedback
    loop = asyncio.get_event_loop()
    g = await loop.run_in_executor(executor, db.grievances.find_one, {"_id": grievance_id})
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    if user["role"] == UserRole.CITIZEN.value and g.get("citizen_user_id") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="You can only provide feedback on your own grievances")
    def update():
        return db.grievances.update_one({"_id": grievance_id},
            {"$set": {"resolution_feedback": feedback.rating, "updated_at": datetime.now(timezone.utc)}})
    result = await loop.run_in_executor(executor, update)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return await get_grievance(grievance_id, user, db)

@app.put("/grievances/{grievance_id}/assign", response_model=GrievanceResponse)
async def assign_grievance(grievance_id: str, assignment: GrievanceAssignment,
                           user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value)),
                           db=Depends(get_db)):
    grievance_id = validate_uuid(grievance_id, "grievance_id")
    def update():
        return db.grievances.update_one({"_id": grievance_id},
            {"$set": {"assigned_officer": assignment.officer_name,
                      "status": GrievanceStatus.IN_PROGRESS.value,
                      "updated_at": datetime.now(timezone.utc)}})
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, update)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return await get_grievance(grievance_id, user, db)

# -- Citizen confirms self-resolution (auth required) --
class ConfirmResolutionRequest(BaseModel):
    tracking_number: str = Field(..., max_length=30)
    helpful: bool

@app.put("/grievances/confirm-resolution")
@limiter.limit("10/minute")
async def confirm_resolution(request: Request, req: ConfirmResolutionRequest,
                             user=Depends(get_current_user), db=Depends(get_db)):
    tracking_number = sanitize_str(req.tracking_number)
    if not re.match(r'^GRV-\d{4}-[A-Za-z0-9]{6,16}$', tracking_number):
        raise HTTPException(status_code=400, detail="Invalid tracking number format")
    loop = asyncio.get_event_loop()
    g = await loop.run_in_executor(executor, db.grievances.find_one,
                                   {"tracking_number": tracking_number})
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    # Verify the current user owns this grievance
    if user["role"] == UserRole.CITIZEN.value and g.get("citizen_user_id") != str(user["_id"]):
        raise HTTPException(status_code=403, detail="You can only confirm resolution on your own grievances")
    if g.get("resolution_tier") != ResolutionTier.SELF_RESOLVABLE.value:
        raise HTTPException(status_code=400, detail="This grievance is not eligible for self-resolution")
    if g.get("status") != GrievanceStatus.PENDING.value:
        raise HTTPException(status_code=400, detail="Grievance is no longer pending")

    if req.helpful:
        update_data = {
            "status": GrievanceStatus.RESOLVED.value,
            "resolution_type": ResolutionType.AI.value,
            "updated_at": datetime.now(timezone.utc),
        }
    else:
        # Promote to officer action
        update_data = {
            "resolution_tier": ResolutionTier.OFFICER_ACTION.value,
            "updated_at": datetime.now(timezone.utc),
        }
    await loop.run_in_executor(executor, lambda: db.grievances.update_one(
        {"_id": g["_id"]}, {"$set": update_data}))
    return {"status": "ok", "new_status": update_data.get("status", g["status"]),
            "new_tier": update_data.get("resolution_tier", g.get("resolution_tier"))}

# -- Officer approves / edits AI draft --
class ApproveDraftRequest(BaseModel):
    resolution: Optional[str] = Field(None, max_length=10000)  # None = use AI draft as-is

@app.put("/grievances/{grievance_id}/approve-draft", response_model=GrievanceResponse)
async def approve_draft(grievance_id: str, req: ApproveDraftRequest,
                        background_tasks: BackgroundTasks,
                        user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value)),
                        db=Depends(get_db)):
    grievance_id = validate_uuid(grievance_id, "grievance_id")
    loop = asyncio.get_event_loop()
    g = await loop.run_in_executor(executor, db.grievances.find_one, {"_id": grievance_id})
    if not g:
        raise HTTPException(status_code=404, detail="Grievance not found")
    final_text = req.resolution if req.resolution else g.get("ai_resolution", "")
    if not final_text:
        raise HTTPException(status_code=400, detail="No resolution text available")
    update_data = {
        "manual_resolution": final_text,
        "resolution_type": ResolutionType.HYBRID.value,
        "status": GrievanceStatus.RESOLVED.value,
        "assigned_officer": user["full_name"],
        "updated_at": datetime.now(timezone.utc),
    }
    # Also add to service memory
    background_tasks.add_task(add_to_service_memory, g, final_text, user["full_name"])
    await loop.run_in_executor(executor, lambda: db.grievances.update_one(
        {"_id": grievance_id}, {"$set": update_data}))
    return await get_grievance(grievance_id, user, db)

# ---------------------------------------------------------------------------
# CHATBOT ENDPOINT (with grievance-filing integration)
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("15/minute")
async def chat(request: Request, msg: ChatMessage, user=Depends(get_current_user), db=Depends(get_db)):
    lang_name = LANGUAGE_NAMES.get(msg.language.value, "English")
    doc_results = await KnowledgeSearcher.search_documentation(msg.message, limit=3)
    memory_results = await KnowledgeSearcher.search_service_memory(msg.message, limit=3)
    scheme_results = await KnowledgeSearcher.search_schemes(msg.message, limit=3)

    context_parts, sources = [], []
    for d in doc_results:
        context_parts.append(f"Documentation: {d['title']} - {d['content']}")
        sources.append({"type": "documentation", "title": d["title"], "score": d["score"]})
    for m in memory_results:
        context_parts.append(f"Past resolution: {m['query']} -> {m['resolution']}")
        sources.append({"type": "service_memory", "query": m["query"], "score": m["score"]})
    for s in scheme_results:
        context_parts.append(f"Scheme: {s['name']} - {s['description']}. Eligibility: {s['eligibility']}")
        sources.append({"type": "scheme", "name": s["name"], "score": s["score"]})
    context_text = "\n".join(context_parts) if context_parts else "No specific knowledge base context found."

    system_prompt = (
        f"You are Seva, a professional government assistant for the Odisha Panchayati Raj & Drinking Water Department Grievance Portal.\n"
        f"The department handles: rural drinking water (Jal Jeevan Mission, Basudha), MGNREGS, rural housing (PMAY-G), "
        f"panchayat governance, rural livelihoods (NRLM/OLM/SHGs), sanitation (SBM-G), and rural infrastructure (BGBO).\n"
        f"Respond in {lang_name}. Use Markdown formatting.\n\n"
        "RESPONSE RULES (strictly follow):\n"
        "- Keep responses under 120 words. Be concise and direct.\n"
        "- Use **bold** for key terms, names of schemes, and important details.\n"
        "- Use bullet points for lists or steps.\n"
        "- Do not repeat what the citizen just said. Go straight to the answer.\n"
        "- One short greeting max (first message only). After that, be direct.\n"
        "- Do not pad with filler phrases like 'I understand your concern' or 'Thank you for reaching out'.\n\n"
        "GRIEVANCE FILING PROTOCOL:\n"
        "When a citizen wants to file/register/submit a grievance, do NOT file immediately.\n"
        "Instead, gather information through conversation. You need AT LEAST:\n"
        "  1. What is the problem? (specific issue, not just a category)\n"
        "  2. Which DISTRICT? (must be one of the 30 Odisha districts listed below)\n"
        "  3. Where exactly? (village/block/GP name, landmark, or address)\n"
        "  4. When / how long? (when it started or was noticed)\n"
        "  5. Any prior complaints or reference numbers?\n\n"
        "ODISHA DISTRICTS (valid values — you MUST identify which one applies):\n"
        f"  {', '.join(ODISHA_DISTRICTS)}\n\n"
        "After each citizen reply, mentally check which items you still need.\n"
        "- If items are missing, ask for the NEXT missing item (one or two at a time, not all at once).\n"
        "- The district is MANDATORY. If the citizen mentions a village/block/area but not the district, "
        "ask them to confirm which district it falls under.\n"
        "- Once you have items 1-4 at minimum, summarize what you have and ask:\n"
        "  'Shall I file this grievance now, or would you like to add anything?'\n"
        "- ONLY when the citizen confirms (says yes/file/submit/go ahead), include this JSON block "
        "at the END of your response:\n"
        '```json\n{"action":"file_grievance","title":"brief title","description":"full details including location, timing, and specifics","district":"ExactDistrictName"}\n```\n'
        "The description in the JSON must be a comprehensive paragraph combining ALL gathered details.\n"
        "The district in the JSON MUST be one of the 30 valid Odisha district names listed above (exact spelling).\n"
        "NEVER include the JSON block before the citizen confirms filing.\n\n"
        f"Knowledge base context:\n{context_text}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    ALLOWED_ROLES = {"user", "assistant"}
    for h in msg.conversation_history[-10:]:
        role = h.get("role", "user")
        if role not in ALLOWED_ROLES:
            role = "user"
        content = str(h.get("content", ""))[:2000]
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": msg.message})

    try:
        reply = await openai_chat(messages)
        if not reply:
            return ChatResponse(reply="I apologize, I'm having trouble right now. Please try again.", sources=[])

        filed_grievance = None
        json_match = re.search(r'```json\s*(\{.*?"action"\s*:\s*"file_grievance".*?\})\s*```', reply, re.DOTALL)
        if json_match:
            try:
                action_data = json.loads(json_match.group(1))
                if action_data.get("action") == "file_grievance" and action_data.get("title"):
                    district_val = action_data.get("district")
                    if district_val and district_val not in ODISHA_DISTRICTS:
                        district_val = None
                    grievance_data = GrievanceCreate(
                        title=action_data["title"],
                        description=action_data.get("description", action_data["title"]),
                        language=msg.language,
                        district=district_val,
                        citizen_name=user.get("full_name"),
                        citizen_email=user.get("email"),
                    )
                    result = await process_grievance(grievance_data, db, user)
                    filed_grievance = {
                        "tracking_number": result.tracking_number,
                        "status": result.status.value,
                        "department": result.department.value,
                        "priority": result.priority.value,
                    }
                    reply = reply[:json_match.start()].strip()
            except Exception as e:
                logger.error("Chat grievance filing error: %s", e)

        return ChatResponse(reply=reply, sources=sources, filed_grievance=filed_grievance)
    except Exception as e:
        logger.error("Chat error: %s", e)
        return ChatResponse(reply="I apologize, I'm having trouble right now. Please try again.", sources=[])

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE ENDPOINTS
# ---------------------------------------------------------------------------
async def add_to_service_memory(grievance: dict, resolution: str, officer: str):
    try:
        query = f"{grievance['title']} {grievance['description']}"
        embedding = await get_openai_embedding(query)
        point = PointStruct(id=str(uuid.uuid4()), vector=embedding,
            payload={"query": query, "resolution": resolution,
                     "category": grievance.get("department", "general"),
                     "agent_name": officer,
                     "tracking_number": grievance.get("tracking_number"),
                     "grievance_id": str(grievance.get("_id", "")),
                     "created_at": datetime.now(timezone.utc).isoformat()})
        qdrant.upsert(collection_name="service_memory", points=[point], wait=True)
    except Exception as e:
        logger.error("Error adding to service memory: %s", e)

@app.post("/knowledge/documentation")
async def add_documentation(entry: KnowledgeEntry,
                            user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    embedding = await get_openai_embedding(entry.content)
    point = PointStruct(id=str(uuid.uuid4()), vector=embedding,
        payload={"title": entry.title, "content": entry.content,
                 "category": entry.category.value, "created_at": datetime.now(timezone.utc).isoformat()})
    qdrant.upsert(collection_name="documentation", points=[point], wait=True)
    return {"message": "Documentation added successfully"}

@app.post("/knowledge/service-memory")
async def add_service_memory_endpoint(entry: ServiceMemoryEntry,
                                       user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    embedding = await get_openai_embedding(entry.query)
    point = PointStruct(id=str(uuid.uuid4()), vector=embedding,
        payload={"query": entry.query, "resolution": entry.resolution,
                 "category": entry.category.value, "agent_name": entry.agent_name,
                 "created_at": datetime.now(timezone.utc).isoformat()})
    qdrant.upsert(collection_name="service_memory", points=[point], wait=True)
    return {"message": "Service memory added successfully"}

@app.post("/knowledge/schemes")
async def add_scheme(entry: SchemeEntry,
                     user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    text = f"{entry.name} {entry.description} {entry.eligibility} {entry.how_to_apply}"
    embedding = await get_openai_embedding(text)
    point = PointStruct(id=str(uuid.uuid4()), vector=embedding,
        payload={"name": entry.name, "description": entry.description,
                 "eligibility": entry.eligibility, "department": entry.department.value,
                 "how_to_apply": entry.how_to_apply,
                 "eligibility_questions": entry.eligibility_questions,
                 "created_at": datetime.now(timezone.utc).isoformat()})
    qdrant.upsert(collection_name="schemes", points=[point], wait=True)
    return {"message": "Scheme added successfully"}

@app.post("/knowledge/schemes/generate-questions")
async def generate_questions_preview(req: GenerateQuestionsRequest,
                                      user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    questions = await generate_eligibility_questions(req.eligibility, req.scheme_name)
    return {"questions": questions}

@app.post("/knowledge/schemes/backfill-questions")
async def backfill_scheme_questions(
    user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    zero_vector = [0.0] * EMBEDDING_DIM
    results = qdrant.query_points(collection_name="schemes", query=zero_vector,
                                   limit=100, with_payload=True, with_vectors=False)
    updated = 0
    for p in results.points:
        existing = p.payload.get("eligibility_questions", [])
        if not existing:
            elig = p.payload.get("eligibility", "")
            name = p.payload.get("name", "")
            if elig:
                questions = await generate_eligibility_questions(elig, name)
                if questions:
                    qdrant.set_payload(collection_name="schemes",
                                       payload={"eligibility_questions": questions},
                                       points=[p.id])
                    updated += 1
    return {"message": f"Backfilled questions for {updated} schemes"}

@app.get("/knowledge/schemes/{scheme_id}", response_model=SchemeResponse)
async def get_scheme_detail(scheme_id: str):
    points = qdrant.retrieve(collection_name="schemes", ids=[scheme_id], with_payload=True)
    if not points:
        raise HTTPException(status_code=404, detail="Scheme not found")
    p = points[0]
    return {"id": p.id, "name": p.payload.get("name", ""), "description": p.payload.get("description", ""),
            "eligibility": p.payload.get("eligibility", ""), "department": p.payload.get("department", ""),
            "how_to_apply": p.payload.get("how_to_apply", ""), "created_at": p.payload.get("created_at", ""),
            "eligibility_questions": p.payload.get("eligibility_questions", [])}

@app.put("/knowledge/schemes/{scheme_id}/questions")
async def update_scheme_questions(scheme_id: str, req: UpdateQuestionsRequest,
                                   user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    points = qdrant.retrieve(collection_name="schemes", ids=[scheme_id], with_payload=True)
    if not points:
        raise HTTPException(status_code=404, detail="Scheme not found")
    qdrant.set_payload(collection_name="schemes",
                       payload={"eligibility_questions": req.questions},
                       points=[scheme_id])
    return {"message": "Questions updated successfully"}

@app.get("/knowledge/documentation", response_model=List[DocumentationResponse])
async def get_documentation(category: Optional[Department] = None,
                            limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0, le=10000),
                            user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    filt = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category.value))]) if category else None
    zero_vector = [0.0] * EMBEDDING_DIM
    results = qdrant.query_points(collection_name="documentation", query=zero_vector,
                            query_filter=filt, limit=limit, offset=offset, with_payload=True, with_vectors=False)
    return [{"id": p.id, "title": p.payload.get("title", ""), "content": p.payload.get("content", ""),
             "category": p.payload.get("category", ""), "created_at": p.payload.get("created_at", "")} for p in results.points]

@app.get("/knowledge/service-memory", response_model=List[ServiceMemoryResponse])
async def get_service_memory(category: Optional[Department] = None,
                             limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0, le=10000),
                             user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    filt = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category.value))]) if category else None
    zero_vector = [0.0] * EMBEDDING_DIM
    results = qdrant.query_points(collection_name="service_memory", query=zero_vector,
                            query_filter=filt, limit=limit, offset=offset, with_payload=True, with_vectors=False)
    return [{"id": p.id, "query": p.payload.get("query", ""), "resolution": p.payload.get("resolution", ""),
             "category": p.payload.get("category", ""), "agent_name": p.payload.get("agent_name", ""),
             "tracking_number": p.payload.get("tracking_number"),
             "grievance_id": p.payload.get("grievance_id"),
             "created_at": p.payload.get("created_at", "")} for p in results.points]

@app.delete("/knowledge/documentation/{doc_id}")
async def delete_documentation(doc_id: str,
                                user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    qdrant.delete(collection_name="documentation", points_selector=PointIdsList(points=[doc_id]))
    return {"message": "Documentation deleted"}

@app.delete("/knowledge/service-memory/{mem_id}")
async def delete_service_memory(mem_id: str,
                                 user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    qdrant.delete(collection_name="service_memory", points_selector=PointIdsList(points=[mem_id]))
    return {"message": "Service memory entry deleted"}

@app.delete("/knowledge/schemes/{scheme_id}")
async def delete_scheme(scheme_id: str,
                         user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    qdrant.delete(collection_name="schemes", points_selector=PointIdsList(points=[scheme_id]))
    return {"message": "Scheme deleted"}

@app.put("/knowledge/schemes/{scheme_id}")
async def update_scheme(scheme_id: str, entry: SchemeEntry,
                         user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value))):
    points = qdrant.retrieve(collection_name="schemes", ids=[scheme_id], with_payload=True)
    if not points:
        raise HTTPException(status_code=404, detail="Scheme not found")
    # Re-embed with updated text
    text = f"{entry.name} {entry.description} {entry.eligibility} {entry.how_to_apply}"
    embedding = await get_openai_embedding(text)
    # Preserve existing questions if not provided in the update
    existing_questions = points[0].payload.get("eligibility_questions", [])
    new_questions = entry.eligibility_questions if entry.eligibility_questions else existing_questions
    point = PointStruct(id=scheme_id, vector=embedding,
        payload={"name": entry.name, "description": entry.description,
                 "eligibility": entry.eligibility, "department": entry.department.value,
                 "how_to_apply": entry.how_to_apply,
                 "eligibility_questions": new_questions,
                 "created_at": points[0].payload.get("created_at", datetime.now(timezone.utc).isoformat())})
    qdrant.upsert(collection_name="schemes", points=[point], wait=True)
    return {"message": "Scheme updated successfully"}

@app.get("/knowledge/schemes", response_model=List[SchemeResponse])
async def get_schemes(department: Optional[Department] = None,
                      limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0, le=10000)):
    filt = Filter(must=[FieldCondition(key="department", match=MatchValue(value=department.value))]) if department else None
    zero_vector = [0.0] * EMBEDDING_DIM
    results = qdrant.query_points(collection_name="schemes", query=zero_vector,
                            query_filter=filt, limit=limit, offset=offset, with_payload=True, with_vectors=False)
    return [{"id": p.id, "name": p.payload.get("name", ""), "description": p.payload.get("description", ""),
             "eligibility": p.payload.get("eligibility", ""), "department": p.payload.get("department", ""),
             "how_to_apply": p.payload.get("how_to_apply", ""), "created_at": p.payload.get("created_at", ""),
             "eligibility_questions": p.payload.get("eligibility_questions", [])} for p in results.points]

@app.post("/knowledge/schemes/search")
@limiter.limit("20/minute")
async def search_schemes_endpoint(request: Request, query: str = Query(...)):
    return await KnowledgeSearcher.search_schemes(query, limit=10)

# ---------------------------------------------------------------------------
# ANALYTICS ENDPOINTS (real computation)
# ---------------------------------------------------------------------------
@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(days: int = Query(30, ge=1, le=365),
                        user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value)),
                        db=Depends(get_db)):
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    def fetch():
        total = db.grievances.count_documents({"created_at": {"$gte": start_date}})
        self_resolved = db.grievances.count_documents({
            "created_at": {"$gte": start_date}, "status": "resolved",
            "resolution_tier": "self_resolvable", "resolution_type": "ai"})
        ai_drafted = db.grievances.count_documents({
            "created_at": {"$gte": start_date},
            "ai_resolution": {"$exists": True, "$ne": None}})
        escalated = db.grievances.count_documents({"created_at": {"$gte": start_date}, "status": "escalated"})
        sent_results = list(db.grievances.aggregate([
            {"$match": {"created_at": {"$gte": start_date}}},
            {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}]))
        dept_results = list(db.grievances.aggregate([
            {"$match": {"created_at": {"$gte": start_date}}},
            {"$group": {"_id": "$department", "count": {"$sum": 1}}}]))
        # Real avg resolution time
        avg_pipe = [
            {"$match": {"created_at": {"$gte": start_date}, "status": "resolved"}},
            {"$project": {"duration": {"$subtract": ["$updated_at", "$created_at"]}}},
            {"$group": {"_id": None, "avg_ms": {"$avg": "$duration"}}}]
        avg_result = list(db.grievances.aggregate(avg_pipe))
        avg_hours = (avg_result[0]["avg_ms"] / 3600000) if avg_result and avg_result[0].get("avg_ms") else 0.0
        # Pending and SLA breached counts
        now = datetime.now(timezone.utc)
        pending_count = db.grievances.count_documents({
            "created_at": {"$gte": start_date},
            "status": {"$in": ["pending", "in_progress", "escalated"]}})
        sla_breached_count = db.grievances.count_documents({
            "created_at": {"$gte": start_date},
            "status": {"$in": ["pending", "in_progress", "escalated"]},
            "sla_deadline": {"$lt": now}})
        # Status distribution
        status_results = list(db.grievances.aggregate([
            {"$match": {"created_at": {"$gte": start_date}}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}]))
        status_dist = {s["_id"]: s["count"] for s in status_results}
        # Top districts by grievance count (all, not capped at 10)
        district_pipe = [
            {"$match": {"created_at": {"$gte": start_date}, "district": {"$ne": None}}},
            {"$group": {"_id": "$district", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}]
        district_results = list(db.grievances.aggregate(district_pipe))
        top_districts = [{"district": d["_id"], "count": d["count"]} for d in district_results]
        # Top complaint titles
        complaint_pipe = [
            {"$match": {"created_at": {"$gte": start_date}}},
            {"$group": {"_id": "$title", "department": {"$first": "$department"}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}, {"$limit": 10}]
        complaint_results = list(db.grievances.aggregate(complaint_pipe))
        top_complaints = [{"title": c["_id"], "department": c.get("department", "general"),
                           "count": c["count"]} for c in complaint_results]
        return {"total": total, "self_resolved": self_resolved, "ai_drafted": ai_drafted,
                "escalated": escalated,
                "sent": sent_results, "dept": dept_results, "avg_hours": avg_hours,
                "pending_count": pending_count, "sla_breached_count": sla_breached_count,
                "status_dist": status_dist,
                "top_districts": top_districts, "top_complaints": top_complaints}
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, fetch)
    return AnalyticsResponse(
        total_grievances=data["total"], self_resolved=data["self_resolved"],
        ai_drafted=data["ai_drafted"],
        escalated_to_human=data["escalated"], avg_resolution_time=round(data["avg_hours"], 1),
        pending_count=data["pending_count"], sla_breached_count=data["sla_breached_count"],
        status_distribution=data["status_dist"],
        sentiment_distribution={i["_id"]: i["count"] for i in data["sent"]},
        department_distribution={i["_id"]: i["count"] for i in data["dept"]},
        top_districts=data["top_districts"],
        top_complaints=data["top_complaints"])

@app.get("/analytics/geographical", response_model=GeoAnalyticsResponse)
async def get_geo_analytics(days: int = Query(30, ge=1, le=365),
                            user=Depends(require_role(UserRole.OFFICER.value, UserRole.ADMIN.value)),
                            db=Depends(get_db)):
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    def fetch():
        hotspots, district_counts = [], {}
        for g in db.grievances.find({"created_at": {"$gte": start_date}, "location": {"$exists": True, "$ne": None}}).limit(200):
            coords = g.get("location", {}).get("coordinates", [])
            if len(coords) == 2:
                hotspots.append({"latitude": coords[1], "longitude": coords[0],
                                 "priority": g.get("priority"), "department": g.get("department")})
            d = g.get("district", "unknown")
            district_counts[d] = district_counts.get(d, 0) + 1
        return {"hotspots": hotspots, "district_counts": district_counts}
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, fetch)
    return GeoAnalyticsResponse(hotspot_coordinates=data["hotspots"],
                                region_distribution={}, district_distribution=data["district_counts"])

# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "system": "PR&DW Grievance Portal",
            "timestamp": datetime.now(timezone.utc)}

# ---------------------------------------------------------------------------
# PAGE ROUTES (serve Jinja2 templates)
# ---------------------------------------------------------------------------
PAGES = [
    ("", "login.html"), ("login", "login.html"), ("register", "register.html"),
    ("dashboard", "dashboard.html"), ("file-grievance", "file_grievance.html"),
    ("track", "track.html"), ("chatbot", "chatbot.html"), ("schemes", "schemes.html"),
    ("officer-dashboard", "officer_dashboard.html"), ("queue", "queue.html"),
    ("grievance-detail", "grievance_detail.html"), ("knowledge", "knowledge.html"),
    ("scheme-detail", "scheme_detail.html"),
    ("analytics-view", "analytics.html"),
    ("admin", "admin.html"),
]

for _path, _template in PAGES:
    def _make_handler(tmpl: str):
        async def handler(request: Request):
            return templates.TemplateResponse(tmpl, {"request": request})
        return handler
    app.add_api_route(f"/{_path}" if _path else "/", _make_handler(_template),
                      methods=["GET"], response_class=HTMLResponse, include_in_schema=False)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
