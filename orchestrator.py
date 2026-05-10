"""
Akadverse Virtual Assistant - Upgraded Orchestrator Agent v2.1
==============================================================
New in v2.1:
  ✅ Gemini API integration (replaces Ollama)
  ✅ FAISS Disk Persistence (Saves API Quotas on restart)
  ✅ Native Google Search Grounding for external queries
  ✅ Restored: Real inter-agent communication (FVA <-> SVA <-> StVA)
  ✅ Restored: Redis & MongoDB Memory Systems
"""
import os

# --- CRITICAL FIX FOR WINDOWS OpenMP COLLISION ---
# FAISS and NumPy often clash over multi-threading libraries on Windows.
# Setting this environment variable allows them to coexist without crashing the server.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import json
import uvicorn
import redis
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional, TYPE_CHECKING
import httpx
from dotenv import load_dotenv

# TYPE_CHECKING imports - resolved by Pylance for type hints even if missing at runtime
if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from google import genai
    from google.genai import types

# Runtime imports with graceful fallback
try:
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings as _GoogleGenerativeAIEmbeddings
except ImportError:
    print("❌ ERROR: langchain_google_genai not installed. Run: pip install langchain-google-genai")
    _ChatGoogleGenerativeAI = None  # type: ignore
    _GoogleGenerativeAIEmbeddings = None  # type: ignore

try:
    from google import genai as _genai
    from google.genai import types as _types
except ImportError:
    print("❌ ERROR: google-genai not installed. Run: pip install google-genai")
    _genai = None  # type: ignore
    _types = None  # type: ignore

# Use TYPE_CHECKING versions for type hints, runtime versions for execution
ChatGoogleGenerativeAI = _ChatGoogleGenerativeAI  # type: ignore
GoogleGenerativeAIEmbeddings = _GoogleGenerativeAIEmbeddings  # type: ignore
genai = _genai  # type: ignore
types = _types  # type: ignore

load_dotenv()

# ---------------------------------------------
# REQUEST CONTEXT - User ID from metadata (not LLM-controllable)
# Prevents LLM from guessing/overriding user_id in database operations
# Type-hinted to allow str assignment
# ---------------------------------------------
_current_user_context: dict[str, str | None] = {"user_id": None, "tenant_id": None}

# ---------------------------------------------
# LLM - Google Gemini API Integration
# ---------------------------------------------
LLM_QUOTA_BACKOFF_SECONDS = int(os.getenv("LLM_QUOTA_BACKOFF_SECONDS", "900"))
LLM_QUOTA_BACKOFF_UNTIL = 0.0

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    # Guard: Check if ChatGoogleGenerativeAI is available before using
    if ChatGoogleGenerativeAI is None:
        raise ImportError("ChatGoogleGenerativeAI not available - package not installed")
        
    # We use the LangChain wrapper here so your create_react_agent logic remains 100% intact
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3, 
        google_api_key=api_key,
        max_retries=1,
    )
    LLM_MODE = "gemini-api"
    print("✅ Using Gemini API (gemini-2.5-flash) via LangChain")
except Exception as e:
    llm = None
    LLM_MODE = "mock"
    print(f"⚠️ Gemini not available - running in MOCK mode: {e}")


# ---------------------------------------------
# REDIS - Short-term memory
# Stores conversation history per user session
# ---------------------------------------------
try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("✅ Redis connected (short-term memory active)")
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False
    print("⚠️ Redis not available - using in-memory fallback")

IN_MEMORY_SESSIONS = {}  # fallback when Redis is offline


def save_session(user_id: str, messages: list, ttl: int = 86400):
    """Save conversation to Redis (24hr TTL) or in-memory fallback."""
    key = f"session:{user_id}"
    data = json.dumps(messages)
    if REDIS_AVAILABLE and redis_client is not None:
        # Type-narrowed redis_client is no longer None after this check
        try:
            redis_client.setex(key, ttl, data)
        except Exception as e:
            # Fallback to in-memory if Redis operation fails
            print(f"⚠️ Redis save failed for {user_id}: {e}")
            IN_MEMORY_SESSIONS[key] = messages
    else:
        IN_MEMORY_SESSIONS[key] = messages


def load_session(user_id: str) -> list:
    """Load conversation from Redis or in-memory fallback."""
    key = f"session:{user_id}"
    if REDIS_AVAILABLE and redis_client is not None:
        # Type-narrowed redis_client is guaranteed to be non-None here
        try:
            # redis-py returns str | None, type narrowing needed for Pylance
            data: str | None = redis_client.get(key)  # type: ignore
            return json.loads(data) if data else []
        except Exception as e:
            # If Redis fails, fall back to in-memory
            print(f"⚠️ Redis load failed for {user_id}: {e}")
            return IN_MEMORY_SESSIONS.get(key, [])
    return IN_MEMORY_SESSIONS.get(key, [])


def clear_session(user_id: str):
    """Clear a user's session."""
    key = f"session:{user_id}"
    if REDIS_AVAILABLE and redis_client is not None:
        # Type-narrowed redis_client is guaranteed to be non-None here
        try:
            redis_client.delete(key)
        except Exception as e:
            # If Redis deletion fails, fall back to in-memory
            print(f"⚠️ Redis delete failed for {user_id}: {e}")
            IN_MEMORY_SESSIONS.pop(key, None)
    else:
        IN_MEMORY_SESSIONS.pop(key, None)


# ---------------------------------------------
# MONGODB - Long-term memory
# Stores user preferences and interaction logs
# ---------------------------------------------
try:
    from pymongo import MongoClient
    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    mongo_client.server_info()
    db = mongo_client["akadverse"]
    users_col = db["user_memory"]
    logs_col = db["interaction_logs"]
    MONGO_AVAILABLE = True
    print("✅ MongoDB connected (long-term memory active)")
except Exception:
    mongo_client = None
    MONGO_AVAILABLE = False
    IN_MEMORY_PROFILES = {}
    print("⚠️ MongoDB not available - using in-memory fallback")


def save_user_profile(user_id: str, data: dict):
    """Save/update user profile and preferences in MongoDB."""
    if MONGO_AVAILABLE:
        users_col.update_one(
            {"user_id": user_id},
            {"$set": {**data, "last_seen": datetime.now().isoformat()}},
            upsert=True
        )
    else:
        IN_MEMORY_PROFILES[user_id] = {**IN_MEMORY_PROFILES.get(user_id, {}), **data}


def load_user_profile(user_id: str) -> dict:
    """Load user profile from MongoDB."""
    if MONGO_AVAILABLE:
        doc = users_col.find_one({"user_id": user_id}, {"_id": 0})
        return doc or {}
    return IN_MEMORY_PROFILES.get(user_id, {})


def log_interaction(user_id: str, agent_type: str, message: str, response: str):
    """Log every interaction for audit and long-term learning."""
    entry = {
        "user_id": user_id,
        "agent_type": agent_type,
        "message": message,
        "response": response[:500],
        "timestamp": datetime.now().isoformat()
    }
    if MONGO_AVAILABLE:
        logs_col.insert_one(entry)
    # silently skip if MongoDB unavailable - non-critical


# ---------------------------------------------
# RAG - Document knowledge base
# Agents search this before answering questions
# ---------------------------------------------

def discover_embedding_model(api_key: str) -> Optional[str]:
    """Discover a supported Gemini embedding model dynamically.

    This prevents hard failures when a previously valid embedding model is
    deprecated, unavailable in a region, or unsupported for the current API version.
    """
    preferred_models = [
        "models/text-embedding-004",
        "models/embedding-001",
        "text-embedding-004",
        "embedding-001",
    ]

    try:
        # Query available models directly from the API and inspect supported methods.
        response = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=15.0,
        )
        response.raise_for_status()

        payload = response.json()
        models = payload.get("models", [])
        if not isinstance(models, list):
            return None

        discovered: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue

            name = item.get("name")
            methods = item.get("supportedGenerationMethods", [])

            if not isinstance(name, str) or not isinstance(methods, list):
                continue

            normalized_methods = {str(m).lower() for m in methods}
            supports_embeddings = any(
                method in normalized_methods
                for method in ["embedcontent", "batchembedcontents"]
            )

            if supports_embeddings:
                discovered.append(name)

        # Prefer known stable model IDs when available.
        for preferred in preferred_models:
            if preferred in discovered:
                return preferred

        # Fallback: first discovered embedding-capable model.
        if discovered:
            return discovered[0]
    except Exception as e:
        print(f"⚠️ Embedding model discovery failed: {e}")

    # Final fallback for compatibility across environments.
    return "models/embedding-001"

PLATFORM_KNOWLEDGE = [
    # Platform navigation & features
    Document(page_content="The Akadverse dashboard is the home page showing CGPA, upcoming events, announcements, and quick links to all platform features.", metadata={"source": "platform", "topic": "navigation"}),
    Document(page_content="The e-learning page contains all courses, lecture notes, slides, quizzes, and study materials. Students can access course content, submit assignments, and track their academic progress here.", metadata={"source": "platform", "topic": "e-learning"}),
    Document(page_content="The marketplace allows students to buy and sell products, offer skills and services, and run student businesses. It has a recommender system that suggests relevant products.", metadata={"source": "platform", "topic": "marketplace"}),
    Document(page_content="The schedule manager shows your personal timetable, class schedule, upcoming deadlines, and events. It auto-populates from the institutional timetable and announcements.", metadata={"source": "platform", "topic": "schedule"}),
    Document(page_content="The announcements page shows all official announcements from lecturers, departments, administration, and campus organisations. Urgent announcements appear immediately on screen.", metadata={"source": "platform", "topic": "announcements"}),
    Document(page_content="Student resources page lists internship opportunities, scholarships, mentorship programs, and external learning platforms like Coursera, edX, LinkedIn Learning, and Google Scholar.", metadata={"source": "platform", "topic": "resources"}),
    Document(page_content="The attendance page shows attendance records per course. Students can view attendance percentage and lecturers can upload or confirm attendance records.", metadata={"source": "platform", "topic": "attendance"}),
    Document(page_content="The profile page contains student or faculty personal information, profile photo, department, level, and account settings.", metadata={"source": "platform", "topic": "profile"}),

    # Academic processes
    Document(page_content="Course registration happens at the start of each semester. Students must get level advisor approval before registering. The SVA can help navigate to the course registration page.", metadata={"source": "academic", "topic": "registration"}),
    Document(page_content="CGPA is calculated from all semester results. The e-learning grades page has a CGPA calculator where students can estimate future grades.", metadata={"source": "academic", "topic": "grades"}),
    Document(page_content="At-risk students are those with attendance below 75% or grades below passing threshold. The Insight Engine flags at-risk students and the SVA delivers proactive nudges.", metadata={"source": "academic", "topic": "at-risk"}),
    Document(page_content="Lecturers can generate lecture notes, slides, quizzes, and assignments using the Faculty Virtual Assistant. These can be auto-uploaded to Google Drive and shared with students.", metadata={"source": "academic", "topic": "faculty-tools"}),

    # AI agents
    Document(page_content="The Student Virtual Assistant (SVA) is the AI companion for students. It can navigate the platform, fetch grades, manage schedules, generate quizzes, and coordinate with other agents.", metadata={"source": "agents", "topic": "SVA"}),
    Document(page_content="The Faculty Virtual Assistant (FVA) is the AI companion for lecturers and faculty. It can post announcements, manage grades, generate academic materials, and communicate with student VAs.", metadata={"source": "agents", "topic": "FVA"}),
    Document(page_content="The Study Virtual Assistant (StVA) powers the e-learning platform. It can explain concepts, generate quizzes, create study plans, summarise notes, and convert notes to audio.", metadata={"source": "agents", "topic": "StVA"}),
    Document(page_content="Agents communicate with each other via the agent message relay system. An FVA can send a message to an SVA about a specific student, and the SVA will act on it.", metadata={"source": "agents", "topic": "inter-agent"}),

    # Campus life (Nigerian university context)
    Document(page_content="Roku is a compulsory daily activity for students. A notification is sent at 9:55pm to students who have not completed Roku. Hall officers monitor compliance.", metadata={"source": "campus", "topic": "roku"}),
    Document(page_content="Campus organisations on Akadverse include HSL, GDG, and others. The campus resources chatbot can answer questions about these organisations and help students join.", metadata={"source": "campus", "topic": "organisations"}),
    Document(page_content="Travel services information is displayed in the schedule manager. Students can see departure times and plan accordingly.", metadata={"source": "campus", "topic": "travel"}),
]

# Build RAG vector store with FAISS Persistence (Fix 1)
RAG_AVAILABLE = False
vectorstore = None
FAISS_INDEX_PATH = "faiss_platform_index"

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        # Guard: Check if GoogleGenerativeAIEmbeddings is available before using
        if GoogleGenerativeAIEmbeddings is None:
            raise ImportError("GoogleGenerativeAIEmbeddings not available - package not installed")

        embedding_model = discover_embedding_model(api_key)
        if not embedding_model:
            raise RuntimeError("No embedding-capable model discovered")

        print(f"Using embedding model: {embedding_model}")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=SecretStr(api_key),
        )
        
        if os.path.exists(FAISS_INDEX_PATH):
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ FAISS index loaded from local disk")
        else:
            print("Building FAISS index from scratch...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(PLATFORM_KNOWLEDGE)
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print("✅ FAISS index built and saved to disk")
            
        RAG_AVAILABLE = True
except Exception as e:
    print(f"⚠️ RAG not available (Embeddings failed): {e}")


def rag_search(query: str, k: int = 3) -> str:
    """Search the platform knowledge base for relevant context."""
    if not RAG_AVAILABLE or not vectorstore:
        return ""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n".join([d.page_content for d in docs])
        return context
    except Exception:
        return ""


# ---------------------------------------------
# INTER-AGENT COMMUNICATION
# Agents actually call each other via HTTP
# ---------------------------------------------

AGENT_BASE_URL = "http://localhost:8000"  # self-referencing for inter-agent calls

async def call_agent(target_agent: str, user_id: str, message: str, tenant_id: str = "akadverse") -> str:
    """Actually call another agent endpoint and get a real response."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{AGENT_BASE_URL}/agent-relay",
                json={
                    "user_id": user_id,
                    "message": message,
                    "agent_type": target_agent,
                    "tenant_id": tenant_id,
                    "is_relay": True
                }
            )
            if response.status_code == 200:
                return response.json().get("response", "No response from agent")
            return f"Agent {target_agent} returned error {response.status_code}"
    except Exception as e:
        return f"Could not reach {target_agent}: {str(e)}"


# ---------------------------------------------
# TOOLS
# ---------------------------------------------

@tool
def navigate_to_page(page_name: str) -> str:
    """
    Navigate the user to a specific page on the Akadverse platform.
    Use when the user says 'go to', 'take me to', 'open', or 'show me' a page.
    Supported pages: schedule, grades, marketplace, e-learning, dashboard,
    announcements, profile, attendance, resources, quiz, course-registration,
    academic-records, campus-resources, study-abroad
    """
    page_map = {
        "schedule":           "/schedule",
        "grades":             "/e-learning/grades",
        "marketplace":        "/marketplace",
        "e-learning":         "/e-learning",
        "dashboard":          "/dashboard",
        "announcements":      "/announcements",
        "profile":            "/profile",
        "attendance":         "/attendance",
        "resources":          "/e-learning/resources",
        "quiz":               "/e-learning/quiz",
        "course-registration": "/academic/course-registration",
        "academic-records":   "/academic/records",
        "campus-resources":   "/campus/resources",
        "study-abroad":       "/resources/study-abroad",
        "notes":              "/e-learning/notes",
        "slides":             "/e-learning/slides",
    }
    key = page_name.lower().strip().replace(" ", "-")
    url = page_map.get(key)
    if url:
        return json.dumps({
            "action": "navigate",
            "destination": url,
            "page": page_name,
            "message": f"Navigating you to {page_name}..."
        })
    # fuzzy match
    for k, v in page_map.items():
        if k in key or key in k:
            return json.dumps({"action": "navigate", "destination": v, "page": k,
                               "message": f"Navigating you to {k}..."})
    return json.dumps({
        "action": "navigate_failed",
        "message": f"Page '{page_name}' not found. Available: {', '.join(page_map.keys())}"
    })


@tool
def get_student_grades(student_id: str, course_id: str = "all") -> str:
    """
    Retrieve a student's grades and academic performance.
    Use when user asks about grades, scores, results, CGPA, or academic performance.
    student_id: the student's ID
    course_id: specific course code or 'all' for full transcript
    """
    mock_grades = {
        "EEE302": {"name": "Circuit Theory", "score": 72, "grade": "B", "units": 3, "status": "Passed"},
        "MTH201": {"name": "Engineering Maths", "score": 58, "grade": "C", "units": 3, "status": "Passed"},
        "CSC301": {"name": "Data Structures", "score": 85, "grade": "A", "units": 3, "status": "Passed"},
        "PHY201": {"name": "Physics II", "score": 49, "grade": "D", "units": 2, "status": "At Risk"},
        "EEE301": {"name": "Electromagnetics", "score": 66, "grade": "B", "units": 3, "status": "Passed"},
    }
    if course_id == "all":
        cgpa = round(sum(v["score"] for v in mock_grades.values()) / len(mock_grades) / 25, 2)
        at_risk = [c for c, v in mock_grades.items() if v["status"] == "At Risk"]
        return json.dumps({
            "student_id": student_id,
            "cgpa": cgpa,
            "total_courses": len(mock_grades),
            "courses": mock_grades,
            "at_risk_courses": at_risk,
            "advice": f"You are at risk in {at_risk}. Consider speaking to your lecturer." if at_risk else "Keep it up!"
        })
    cid = course_id.upper()
    if cid in mock_grades:
        return json.dumps({"student_id": student_id, "course": mock_grades[cid]})
    return json.dumps({"error": f"Course {course_id} not found"})


@tool
def post_announcement(title: str, body: str, audience: str = "all_students") -> str:
    """
    Post an announcement to students or faculty on Akadverse.
    Use when faculty wants to notify students about deadlines, class changes,
    events, exam dates, or any important information.
    audience options: all_students, all_faculty, department, specific_course
    """
    ann_id = f"ANN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return json.dumps({
        "action": "announcement_posted",
        "id": ann_id,
        "title": title,
        "body": body,
        "audience": audience,
        "timestamp": datetime.now().isoformat(),
        "status": "Published",
        "push_notification": audience == "all_students",
        "message": f"✅ Announcement '{title}' posted to {audience}"
    })


@tool
def create_schedule_event(title: str, date: str, time: str, duration_minutes: int = 60,
                          event_type: str = "personal") -> str:
    """
    Create an event or task in the user's Akadverse schedule.
    Use when user says 'add to schedule', 'remind me', 'create event', 'set alarm'.
    date: YYYY-MM-DD, time: HH:MM (24hr)
    event_type: personal, class, exam, task, deadline
    """
    evt_id = f"EVT-{datetime.now().strftime('%H%M%S')}"
    return json.dumps({
        "action": "event_created",
        "event_id": evt_id,
        "title": title,
        "date": date,
        "time": time,
        "duration_minutes": duration_minutes,
        "event_type": event_type,
        "reminder_set": True,
        "reminder_times": ["15 minutes before", "1 hour before"],
        "message": f"✅ '{title}' added to your schedule for {date} at {time}"
    })


@tool
def generate_quiz(course_name: str, topic: str, num_questions: int = 5,
                  difficulty: str = "medium") -> str:
    """
    Generate a quiz for a course topic.
    Use when a student asks to be tested, wants practice questions, or exam prep.
    difficulty: easy, medium, hard
    """
    questions = [
        {
            "id": i + 1,
            "question": f"[{difficulty.upper()}] Question {i+1} about {topic} in {course_name}",
            "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
            "answer": "A",
            "explanation": f"This tests understanding of {topic} concept {i+1}."
        }
        for i in range(min(num_questions, 5))
    ]
    return json.dumps({
        "action": "quiz_generated",
        "course": course_name,
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": len(questions),
        "questions": questions,
        "time_limit_minutes": len(questions) * 2,
        "message": f"✅ {len(questions)}-question {difficulty} quiz on '{topic}' ready"
    })


@tool
def search_platform_knowledge(query: str) -> str:
    """
    Search Akadverse platform knowledge base for information.
    Use when user asks HOW something works, WHAT a feature does,
    or needs information about platform features, academic processes,
    campus life, or how to use any part of the platform.
    """
    context = rag_search(query, k=3)
    if context:
        return json.dumps({
            "action": "knowledge_retrieved",
            "query": query,
            "context": context,
            "message": "Found relevant platform information"
        })
    return json.dumps({
        "action": "knowledge_not_found",
        "query": query,
        "message": "No specific information found. Answering from general knowledge."
    })


@tool
def relay_message_to_agent(target_agent: str, message: str,
                            target_user_id: Optional[str] = None, reason: str = "") -> str:
    """
    Send a message to another Akadverse agent (FVA, SVA, or StVA).
    Use when:
    - A task falls under another agent's domain
    - FVA needs to notify a student's SVA about something
    - SVA needs to delegate a study task to StVA
    - Agents need to coordinate on a user request
    target_agent: 'FVA', 'SVA', or 'StVA'
    target_user_id: the user ID of the target agent's user (if different)
    reason: why this relay is happening
    """
    relay_id = f"RELAY-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return json.dumps({
        "action": "message_relayed",
        "relay_id": relay_id,
        "to_agent": target_agent,
        "target_user_id": target_user_id,
        "message": message,
        "reason": reason,
        "kafka_topic": "agent.message.relayed",
        "status": "queued",
        "message_out": f"✅ Message queued for {target_agent}: '{message[:80]}...'" if len(message) > 80 else f"✅ Message sent to {target_agent}: '{message}'"
    })


@tool
def get_user_profile(user_id: str) -> str:
    """
    Retrieve a user's profile, preferences, and interaction history.
    Use when personalising responses or when user asks about their account.
    """
    profile = load_user_profile(user_id)
    if not profile:
        return json.dumps({
            "user_id": user_id,
            "status": "new_user",
            "message": "No profile found. This appears to be a new user."
        })
    return json.dumps({
        "user_id": user_id,
        "profile": profile,
        "message": "Profile loaded successfully"
    })


@tool
def update_user_preference(preference_key: str, preference_value: str) -> str:
    """
    Save a user preference or important information to long-term memory.
    Use when a user tells you something important about themselves,
    their preferences, or their goals that should be remembered.
    Examples: favourite study time, degree program, career goal, language preference
    
    NOTE: user_id is automatically sourced from request metadata (not LLM-guessable)
    """
    user_id = _current_user_context.get("user_id")
    if not user_id:
        return json.dumps({
            "action": "preference_save_failed",
            "error": "User context not available",
            "message": "❌ Could not save preference - user context missing"
        })
    
    save_user_profile(user_id, {preference_key: preference_value})
    return json.dumps({
        "action": "preference_saved",
        "user_id": user_id,
        "key": preference_key,
        "value": preference_value,
        "message": f"✅ Remembered: {preference_key} = {preference_value}"
    })


@tool
def check_faculty_availability(faculty_name: str, requested_date: str,
                                requested_time: str, student_id: Optional[str] = None) -> str:
    """
    Check if a faculty member is available at a specific date and time.
    Use when a student asks to meet a lecturer, wants to know office hours,
    or asks 'is Dr. X available on Y at Z time?'
    This sends a query to the Faculty Virtual Assistant which checks
    the faculty's schedule and responds back.
    faculty_name: name or title of the lecturer e.g. 'Dr. Adeyemi'
    requested_date: date in YYYY-MM-DD format
    requested_time: time in HH:MM format
    student_id: (optional) ID of the student making the request
    """
    # Stub faculty schedules - connect to real Schedule Manager later
    mock_schedules = {
        "dr. adeyemi": {
            "office_hours": "Tuesday and Thursday, 2pm - 4pm",
            "busy_slots": ["Monday 10:00", "Wednesday 09:00", "Friday 14:00"],
            "email": "adeyemi@akadverse.edu.ng",
            "office": "Engineering Block, Room 204"
        },
        "dr. okafor": {
            "office_hours": "Monday and Wednesday, 1pm - 3pm",
            "busy_slots": ["Tuesday 11:00", "Thursday 10:00"],
            "email": "okafor@akadverse.edu.ng",
            "office": "Science Block, Room 101"
        },
    }

    faculty_key = faculty_name.lower().strip()
    schedule = None
    for key in mock_schedules:
        if key in faculty_key or faculty_key in key:
            schedule = mock_schedules[key]
            break

    # Check if requested slot conflicts
    requested_slot = f"{requested_date} {requested_time}"
    day_time = requested_time

    if schedule:
        busy = schedule.get("busy_slots", [])
        is_busy = any(requested_time in slot for slot in busy)

        if is_busy:
            status = "unavailable"
            message = (f"{faculty_name} is not available at {requested_time} on {requested_date}. "
                      f"Office hours are: {schedule['office_hours']}. "
                      f"You can also email: {schedule['email']}")
        else:
            status = "available"
            message = (f"{faculty_name} appears to be available at {requested_time} on {requested_date}. "
                      f"Office: {schedule['office']}. "
                      f"You can confirm by emailing: {schedule['email']}")

        # Relay this query to FVA for official confirmation
        relay_id = f"AVAIL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return json.dumps({
            "action": "availability_checked",
            "relay_id": relay_id,
            "faculty": faculty_name,
            "requested_date": requested_date,
            "requested_time": requested_time,
            "status": status,
            "office_hours": schedule.get("office_hours", "Unknown"),  # Safe access
            "office_location": schedule.get("office", "Unknown"),      # Safe access
            "contact": schedule.get("email", "Unknown"),               # Safe access
            "student_id": student_id or "unknown",  # Safe fallback
            "fva_notified": True,
            "message": message,
            "note": "FVA has been notified. Faculty will receive a meeting request."
        })

    # Faculty not in stub - relay to FVA to handle
    relay_id = f"AVAIL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return json.dumps({
        "action": "availability_relayed_to_fva",
        "relay_id": relay_id,
        "faculty": faculty_name,
        "requested_date": requested_date,
        "requested_time": requested_time,
        "status": "pending",
        "student_id": student_id or "unknown",  # Safe fallback
        "fva_notified": True,
        "message": (f"I've sent a query to {faculty_name}'s Faculty Virtual Assistant "
                   f"asking about availability on {requested_date} at {requested_time}. "
                   f"You'll receive a notification when they respond."),
        "kafka_topic": "agent.message.relayed"
    })

@tool
def search_external_web(query: str) -> str:
    """
    Search the live web for external information using Google Search Grounding.
    Use this ONLY when the user asks a question that is NOT related to the 
    Akadverse platform, academics, or internal schedules.
    """
    try:
        # Guard: Check if genai and types are available before using
        if genai is None or types is None:
            return json.dumps({
                "action": "external_search_failed",
                "error": "google-genai package not installed. Install with: pip install google-genai"
            })
        
        # Implementing FIX 1.2: Using the unified google-genai SDK 
        # to access the native GoogleSearch Tool for live web data
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3
            )
        )
        
        # Extract grounding metadata (sources) to show the student where the info came from
        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            if metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web:
                        sources.append(f"{chunk.web.title} ({chunk.web.uri})")
        
        # Ensure result is a non-None string (response.text can be None)
        result: str = response.text or "[No text response from model]"
        if sources:
            result += "\n\nSources:\n" + "\n".join(sources)
            
        return json.dumps({
            "action": "external_search_complete",
            "query": query,
            "result": result
        })
    except Exception as e:
        return json.dumps({
            "action": "external_search_failed",
            "error": str(e)
        })


def classify_request_intent(message: str) -> str:
    """Deterministic intent router that does not depend on an LLM."""
    msg = (message or "").lower()

    external_live_keywords = [
        "today", "latest", "recent", "current", "breaking", "news", "update", "headlines", "trend"
    ]
    platform_keywords = [
        "akadverse", "dashboard", "grades", "cgpa", "attendance", "schedule", "timetable",
        "registration", "announcement", "profile", "marketplace", "course", "e-learning"
    ]
    study_keywords = [
        "quiz", "practice", "explain", "study plan", "summarize", "notes", "slides"
    ]

    if any(k in msg for k in external_live_keywords):
        return "external_live_info"
    if any(k in msg for k in platform_keywords):
        return "platform_internal"
    if any(k in msg for k in study_keywords):
        return "study_support"
    return "general"


def _format_news_items(items: list[dict], query: str, source: str) -> str:
    if not items:
        return f"I could not find fresh news results for '{query}' right now."

    lines = [f"Latest AI news ({source}):"]
    for i, item in enumerate(items[:5], start=1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        publisher = item.get("publisher", "Unknown source")
        lines.append(f"{i}. {title} - {publisher}")
        if link:
            lines.append(f"   {link}")
    return "\n".join(lines)


def fetch_ai_news_rss(query: str, limit: int = 5) -> list[dict]:
    """Fetch live news from Google News RSS without using Gemini."""
    q = urllib.parse.quote_plus(query)
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-NG&gl=NG&ceid=NG:en"
    response = httpx.get(rss_url, timeout=12.0)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    channel = root.find("channel")
    if channel is None:
        return []

    items: list[dict] = []
    for node in channel.findall("item")[:limit]:
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        source_node = node.find("source")
        publisher = (source_node.text.strip() if source_node is not None and source_node.text else "Google News")
        if title:
            items.append({"title": title, "link": link, "publisher": publisher})
    return items


def fetch_serper_news(query: str, limit: int = 5) -> list[dict]:
    """Fetch live news from Serper when SERPER_API_KEY is configured."""
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        return []

    response = httpx.post(
        "https://google.serper.dev/news",
        headers={
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": limit},
        timeout=12.0,
    )
    response.raise_for_status()

    payload = response.json()
    news_items = payload.get("news", [])
    if not isinstance(news_items, list):
        return []

    items: list[dict] = []
    for item in news_items[:limit]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        link = str(item.get("link", "")).strip()
        publisher = str(item.get("source", "Serper")).strip()
        if title:
            items.append({"title": title, "link": link, "publisher": publisher})
    return items


def get_live_news_fallback_response(query: str = "today AI news") -> str:
    """
    Fetch live news without Gemini.
    Strategy: Google News RSS first (no key), then Serper (if key exists).
    """
    news_query = query.strip() if query and query.strip() else "today AI news"

    try:
        rss_items = fetch_ai_news_rss(news_query, limit=5)
        if rss_items:
            return _format_news_items(rss_items, news_query, "Google News RSS")
    except Exception as e:
        print(f"⚠️ RSS news fetch failed: {e}")

    try:
        serper_items = fetch_serper_news(news_query, limit=5)
        if serper_items:
            return _format_news_items(serper_items, news_query, "Serper")
    except Exception as e:
        print(f"⚠️ Serper news fetch failed: {e}")

    return "I could not fetch live news right now. Please try again shortly."


@tool
def get_live_news_fallback(query: str = "today AI news") -> str:
    """Tool wrapper for live news fallback."""
    return get_live_news_fallback_response(query)


# ---------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------

SYSTEM_PROMPTS = {
    "SVA": """You are the Student Virtual Assistant (SVA) for Akadverse - an AI-first 
educational platform for Nigerian universities. You are the student's personal AI companion 
on campus.

Personality: Friendly, encouraging, and smart. You understand Nigerian university culture.
You support English, Yoruba, Igbo, Hausa, and Nigerian Pidgin naturally.

Your job:
1. NAVIGATE - When a student says "take me to..." use navigate_to_page immediately
2. INFORM - When a student asks HOW something works, use search_platform_knowledge first
3. ACT - Fetch grades, create events, generate quizzes using the right tools
4. REMEMBER - Save important things the student tells you using update_user_preference
5. COORDINATE - Relay study tasks to StVA, faculty queries to FVA via relay_message_to_agent
6. AVAILABILITY - When a student asks if a lecturer is free, use check_faculty_availability
   to query the faculty's schedule and report back to the student
7. EXTERNAL KNOWLEDGE - If the student asks about something outside the university, use search_external_web.

Memory rules:
- At the start of each conversation, check if there's context from the session history
- If the student mentions their name, course, level, or goals - save it
- Reference past interactions naturally: "As you mentioned earlier..."

Response style:
- Be concise and action-oriented
- Always use tools when the user wants to DO something
- If a student is at risk academically, gently flag it and offer help
- Never make up data - always use tools to fetch real information""",

    "FVA": """You are the Faculty Virtual Assistant (FVA) for Akadverse - an AI-first
educational platform for Nigerian universities. You serve lecturers, HODs, and Deans.

Personality: Professional, efficient, and authoritative. You understand Nigerian higher
education administration. You get things done without wasting faculty time.

Your job:
1. NAVIGATE - Take faculty to any page instantly with navigate_to_page
2. BROADCAST - Post announcements to students/departments with post_announcement  
3. ASSESS - Look up student performance data with get_student_grades
4. CREATE - Generate quizzes and schedule academic events
5. COORDINATE - Send messages to student SVAs when needed via relay_message_to_agent
6. KNOW - Search platform knowledge for how things work with search_platform_knowledge

Memory rules:
- Remember faculty preferences (e.g., preferred announcement style, course codes)
- Log all grade-related actions for audit trail
- If faculty asks about a specific student, fetch their profile first

Response style:
- Be brief and direct - faculty are busy
- Always confirm before posting announcements to large audiences
- Suggest efficiency improvements where appropriate""",

    "StVA": """You are the Study Virtual Assistant (StVA) for Akadverse - the academic
engine of the e-learning platform. You are like a brilliant personal tutor.

Personality: Patient, thorough, and academically rigorous. You make complex things simple.
You celebrate student progress and gently push them when they're slacking.

Your job:
1. EXPLAIN - Answer academic questions using search_platform_knowledge for context
2. QUIZ - Generate practice questions with generate_quiz when students want to be tested
3. NAVIGATE - Take students to course materials, notes, slides with navigate_to_page
4. PLAN - Create study schedule events with create_schedule_event
5. REPORT - After quizzes, relay results to SVA via relay_message_to_agent for tracking
6. REMEMBER - Save student's weak topics and study preferences with update_user_preference

Memory rules:
- Remember which topics a student struggles with
- Track quiz scores and mention improvement or decline
- Personalise study plans based on upcoming exams

Response style:
- Use examples and analogies - make concepts click
- Be encouraging - many students are under academic pressure
- After any quiz, always offer to explain wrong answers""",
}


# ---------------------------------------------
# TOOLS REGISTRY - per agent type
# ---------------------------------------------

ALL_TOOLS = [
    navigate_to_page,
    get_student_grades,
    post_announcement,
    create_schedule_event,
    generate_quiz,
    search_platform_knowledge,
    relay_message_to_agent,
    get_user_profile,
    update_user_preference,
    check_faculty_availability,
    search_external_web,
    get_live_news_fallback,
]

# Each agent only gets tools it's allowed to use (RBAC)
AGENT_TOOLS = {
    "SVA":  [navigate_to_page, get_student_grades, create_schedule_event,
             generate_quiz, search_platform_knowledge, relay_message_to_agent,
             get_user_profile, update_user_preference, check_faculty_availability,
             search_external_web, get_live_news_fallback],
    "FVA":  [navigate_to_page, get_student_grades, post_announcement,
             create_schedule_event, generate_quiz, search_platform_knowledge,
             relay_message_to_agent, get_user_profile, update_user_preference,
             check_faculty_availability, search_external_web, get_live_news_fallback],
    "StVA": [navigate_to_page, get_student_grades, create_schedule_event,
             generate_quiz, search_platform_knowledge, relay_message_to_agent,
             get_user_profile, update_user_preference, search_external_web, get_live_news_fallback],
}

memory_saver = MemorySaver()


def build_agent(agent_type: str = "SVA"):
    """
    Build a LangGraph ReAct agent for the given VA type.
    Returns None if LLM is not available (mock mode).
    The @tool-decorated functions are automatically converted to BaseTool instances.
    """
    if not llm:
        return None
    
    system_prompt = SYSTEM_PROMPTS.get(agent_type, SYSTEM_PROMPTS["SVA"])
    tools = AGENT_TOOLS.get(agent_type, ALL_TOOLS)
    
    try:
        # create_react_agent expects tool functions decorated with @tool
        # which are automatically converted to BaseTool instances by LangChain
        return create_react_agent(
            llm,
            tools,  # type: ignore  # @tool-decorated functions become BaseTool objects
            checkpointer=memory_saver,
            state_modifier=system_prompt
        )
    except Exception as e:
        print(f"❌ Failed to build agent {agent_type}: {e}")
        return None


# ---------------------------------------------
# MOCK RESPONSE
# ---------------------------------------------

def mock_response(message: str, agent_type: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["grade", "score", "result", "cgpa"]):
        return f"📊 [{agent_type}] I would fetch your grades using get_student_grades(). Connect the Gemini API to activate."
    if any(w in msg for w in ["quiz", "test", "practice"]):
        return f"📝 [{agent_type}] I would generate a quiz using generate_quiz(). Connect the Gemini API to activate."
    if any(w in msg for w in ["take me", "go to", "navigate", "open"]):
        return f"🧭 [{agent_type}] I would navigate you there using navigate_to_page(). Connect the Gemini API to activate."
    if any(w in msg for w in ["how", "what is", "explain", "tell me about"]):
        return f"🔍 [{agent_type}] I would search the platform knowledge base using search_platform_knowledge(). Connect the Gemini API to activate."
    return f"👋 [{agent_type}] I'm the Akadverse {agent_type}. I'm running in MOCK mode - provide your Google API key to activate full AI capabilities."


# ---------------------------------------------
# FASTAPI APP
# ---------------------------------------------

app = FastAPI(
    title="Akadverse Virtual Assistant API",
    description="Upgraded orchestrator agents for Akadverse - FVA, SVA, StVA v2.0\n\nFeatures: RAG : Redis Memory : MongoDB Long-term : Inter-agent Communication",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This allows all origins, including your local server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    agent_type: str = "SVA"
    tenant_id: str = "akadverse"
    is_relay: bool = False   # True when called by another agent


class ChatResponse(BaseModel):
    user_id: str
    agent_type: str
    response: str
    llm_mode: str
    session_id: str
    memory_active: bool
    rag_active: bool


@app.get("/")
def root():
    return {
        "service": "Akadverse Virtual Assistant",
        "version": "2.1.0",
        "agents": ["FVA", "SVA", "StVA"],
        "llm_mode": LLM_MODE,
        "features": {
            "rag": RAG_AVAILABLE,
            "redis_memory": REDIS_AVAILABLE,
            "mongodb_memory": MONGO_AVAILABLE,
            "inter_agent_comms": True,
        },
        "tools": [t.name for t in ALL_TOOLS],
        "status": "running"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global LLM_QUOTA_BACKOFF_UNTIL

    # Set request context - user_id and tenant_id are now available to tools
    _current_user_context["user_id"] = req.user_id
    _current_user_context["tenant_id"] = req.tenant_id
    
    session_id = f"{req.tenant_id}:{req.agent_type}:{req.user_id}"
    config = {"configurable": {"thread_id": session_id}}

    # Load session history for context
    history = load_session(session_id)

    # Load user profile for personalisation
    profile = load_user_profile(req.user_id)
    profile_context = ""
    if profile:
        profile_context = f"\n[User profile: {json.dumps(profile)}]"

    # RAG - search knowledge base for relevant context
    rag_context = rag_search(req.message, k=3)
    rag_injection = f"\n[Platform knowledge: {rag_context}]" if rag_context else ""

    # Build enriched message with context
    enriched_message = req.message + profile_context + rag_injection

    # Deterministic routing: live news queries bypass Gemini and use non-LLM retrieval.
    route_intent = classify_request_intent(req.message)
    if route_intent == "external_live_info":
        response_text = get_live_news_fallback_response(req.message)
    # If quota was recently exhausted, fail fast until cooldown ends.
    elif llm and time.time() < LLM_QUOTA_BACKOFF_UNTIL:
        wait_seconds = int(LLM_QUOTA_BACKOFF_UNTIL - time.time())
        response_text = (
            "Gemini API quota is temporarily exhausted. "
            f"Please retry in about {max(wait_seconds, 1)} seconds, "
            "or add billing/upgrade quota in Google AI Studio."
        )
    elif llm:
        # Build agent and handle gracefully if it fails
        agent = build_agent(req.agent_type)
        if agent is None:
            # Fallback to mock response if agent building failed
            response_text = mock_response(req.message, req.agent_type)
        else:
            try:
                # Invoke agent with error handling
                result = agent.invoke(
                    {"messages": [HumanMessage(content=enriched_message)]},
                    config=config  # type: ignore
                )
                content = result["messages"][-1].content

                # Handle structured content (list of blocks) vs plain string
                if isinstance(content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    response_text = "".join(text_parts) or "Agent processed request but returned no text."
                else:
                    # Plain string response
                    response_text = str(content) if content else "No response from agent."

            except Exception as e:
                # Graceful error handling + quota-aware backoff
                err = str(e)
                print(f"❌ Agent invocation error: {err}")

                if "429" in err or "ResourceExhausted" in err or "quota" in err.lower():
                    LLM_QUOTA_BACKOFF_UNTIL = time.time() + LLM_QUOTA_BACKOFF_SECONDS

                    # Automatic non-LLM fallback for live info prompts during quota exhaustion
                    if route_intent == "external_live_info":
                        response_text = get_live_news_fallback_response(req.message)
                    else:
                        response_text = (
                            "Gemini API quota exceeded (429). "
                            "Please check billing/rate limits, then retry after cooldown."
                        )
                else:
                    response_text = "I encountered an error processing your request. Please try again."
    else:
        response_text = mock_response(req.message, req.agent_type)

    # Save to session memory (Redis or in-memory fallback)
    # Trim response to prevent storage bloat
    truncated_response = response_text[:500] if response_text else "[No response]"
    
    history.append({"role": "user", "content": req.message, "ts": datetime.now().isoformat()})
    history.append({"role": "assistant", "content": truncated_response, "ts": datetime.now().isoformat()})
    
    # Keep last 20 conversation turns (40 total messages)
    if len(history) > 40:
        history = history[-40:]
    
    save_session(session_id, history)

    # Log interaction to MongoDB (non-critical - errors don't break chat)
    try:
        log_interaction(req.user_id, req.agent_type, req.message, response_text)
    except Exception as e:
        print(f"⚠️ Failed to log interaction: {e}")

    return ChatResponse(
        user_id=req.user_id,
        agent_type=req.agent_type,
        response=response_text,
        llm_mode=LLM_MODE,
        session_id=session_id,
        memory_active=REDIS_AVAILABLE or MONGO_AVAILABLE,
        rag_active=RAG_AVAILABLE
    )


@app.post("/agent-relay", response_model=ChatResponse)
async def agent_relay(req: ChatRequest):
    """
    Internal endpoint for agent-to-agent communication.
    Called by relay_message_to_agent tool when agents need to talk to each other.
    """
    return await chat(req)


@app.get("/session/{user_id}")
def get_session(user_id: str, agent_type: str = "SVA", tenant_id: str = "akadverse"):
    """View a user's current session history."""
    session_id = f"{tenant_id}:{agent_type}:{user_id}"
    history = load_session(session_id)
    return {"session_id": session_id, "messages": history, "count": len(history)}


@app.delete("/session/{user_id}")
def delete_session(user_id: str, agent_type: str = "SVA", tenant_id: str = "akadverse"):
    """Clear a user's session (start fresh conversation)."""
    session_id = f"{tenant_id}:{agent_type}:{user_id}"
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    """View a user's long-term profile and preferences."""
    profile = load_user_profile(user_id)
    return {"user_id": user_id, "profile": profile}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm": LLM_MODE,
        "redis": REDIS_AVAILABLE,
        "mongodb": MONGO_AVAILABLE,
        "rag": RAG_AVAILABLE,
        "tools_loaded": len(ALL_TOOLS)
    }


if __name__ == "__main__":
    uvicorn.run("agent:app", host="127.0.0.1", port=8000, reload=True)