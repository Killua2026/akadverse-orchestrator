# AkadVerse: Virtual Assistant Orchestrator

**Tier 1 Agentic AI | Microservice Port: `8000`**

The central intelligence hub of AkadVerse, orchestrating specialized Student, Faculty, and Study VAs with persistent memory and platform-grounded reasoning.

## Table of Contents
- [What This Microservice Does](#what-this-microservice-does)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Getting Your API Key](#getting-your-api-key)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Visual Demo Console](#visual-demo-console)
- [API Endpoints](#api-endpoints)
- [Testing with Swagger UI](#testing-with-swagger-ui)
- [Example Test Inputs](#example-test-inputs)
- [Understanding the Responses](#understanding-the-responses)
- [Generated Files](#generated-files)
- [Common Errors and Fixes](#common-errors-and-fixes)
- [Project Structure](#project-structure)

## What This Microservice Does

This service is a Tier 1 component of the AkadVerse AI-first e-learning platform. It lives in the Platform Core module and serves as the primary entry point for all user-AI interactions.

The Orchestrator manages three distinct personas:

*   **Student Virtual Assistant (SVA):** Focuses on navigation, grade tracking, and personal scheduling.
*   **Faculty Virtual Assistant (FVA):** Manages announcements, student performance audits, and office hours.
*   **Study Virtual Assistant (StVA):** An academic engine for quizzes, explanations, and study planning.

**Core Workflow:**

1.  **Identity Scoping:** Incoming requests are tagged with a non-spoofable `user_id` and `tenant_id` for secure database operations.
2.  **Cognitive Routing:** The agent determines if it can answer from its RAG knowledge base, execute a tool (like checking grades), or perform a live web search.
3.  **Memory Integration:** Conversation history is persisted in Redis for short-term flow, while long-term preferences are stored in MongoDB.

**Key Design Decisions:**

*   **FAISS Disk Persistence:** The platform knowledge base index is saved locally to avoid re-embedding costs on restarts.
*   **Deterministic Routing:** Live news queries bypass the LLM and use RSS/Serper fallbacks to save API quota.
*   **Identity Hardening:** Requests use metadata-sourced user IDs to prevent prompt-injection attacks.

## Architecture Overview
User Request
    |
    v
[API Layer] (FastAPI)
    |
    v
[Identity Guard] (Request-scoped User Context)
    |
    v
[Agent Router] (LangGraph / Gemini 1.5 Flash) <---- [Memory] (Redis/MongoDB)
    |            |            |
    v            v            v
[Tool Registry] [RAG Store] [Web Search]
(40+ Actions)   (FAISS)      (Google Search Tool)
```

| Component      | Technology             | Purpose                      |
| :------------- | :--------------------- | :--------------------------- |
| Orchestrator   | LangGraph + LangChain  | Agentic reasoning and tool-use |
| Primary LLM    | Gemini 1.5 Flash       | Core cognitive engine        |
| Web Search     | Google Search Grounding| Live external information    |
| Memory         | Redis + MongoDB        | Short-term vs long-term storage |
| Knowledge Base | FAISS                  | Semantic search for platform features |

## Prerequisites

*   Python 3.12 or higher
*   Redis (for session memory)
*   MongoDB (for long-term profiles)
*   Google Gemini API Key
*   Windows Users: OpenMP Fix is included in code (`KMP_DUPLICATE_LIB_OK`)

## Getting Your API Key

1.  Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2.  Sign in with your Google account.
3.  Click Create API Key.
4.  Create a `.env` file in your root directory and add: `GOOGLE_API_KEY=your_key_here`

## Installation

**Step 1 -- Set up your project folder**

```bash
mkdir akadverse-orchestrator
cd akadverse-orchestrator
```

**Step 2 -- Create and activate a virtual environment**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

**Step 3 -- Install dependencies**

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| fastapi | Async langchain-google-genai | Gemini integration |
| langgraph | Agentic state management |
| redis | Short-term memory client |
| pymongo | Long-term memory client |
| faiss-cpu | Vector storage |

## Running the Server

```bash
uvicorn agent:app --host 127.0.0.1 --port 8000 --reload
```

Expected Terminal Output:

```
[Startup] AkadVerse Orchestrator initialising...
✅ Using Gemini API (gemini-1.5-flash) via LangChain
short-term memory active)
✅ MongoDB connected (long-term memory active)
✅ FAISS index loaded from local disk
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Visual Demo Console

To see the Orchestrator in action with a polished user interface, you can use the built-in demo console.

1.  Ensure the Python server is running on port 8000.
2.  Host the `akadverse_demo.html` file using a simple local server (to avoid CORS issues).

    ```bash
    python -m http.server 3000
    ```

3.  Open your browser and navigate to:
    [http://127.0.0.1:3000/akadverse_demo.html](http://127.0.0.1:3000/akadverse_demo.html)

**What to test in the Demo:**

*   **Agent Switching:** Use the SVA, FVA, and StVA buttons to switch personas.
*   **Quick Prompts:** Click the suggested prompts to see the agent fetch grades or navigate the platform.
*   **Microservice Health:** The sidebar will show real-time connectivity status for other Tier 5 tools.

## API Endpoints

### 1. `POST /chat`

**What it does:** Primary interface for all Virtual Assistant interactions.

**Request Body:**

| Field | Required | Default | Description |
|---|---|---|---|
| `user_id` | Yes | - | Unique student/staff identifier |
| `message` | Yes | - | The user's query |
| `agent_type` | No | SVA | SVA, FVA, or StVA |

**Success Response (200 OK):**

```json
{
  "user_id": "23CE034397",
  "response": "Hello! I've checked your schedule and you have no classes today.",
  "rag_active": true
}
```

### 2. `GET /health`

**What it does:** Returns status of all sub-systems (LLM, Redis, Mongo, RAG).

## Testing with Swagger UI

With the server running, open:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Example Test Inputs

**Test 1 -- Platform Knowledge (RAG)**

Query: "How does the attendance system work?"
Expected: Agent uses `search_platform_knowledge` and explains the 75% threshold.

**Test 2 -- Tool Use (Navigation)**

Query: "Take me to the marketplace."
Expected: JSON response contains `action: navigate` and `destination: /marketplace`.

**Test 3 -- Agent Relay**

Query: "Ask my lecturer if I can meet him."
Expected: SVA uses `check_faculty_availability` to relay a message to the FVA.

## Understanding the Responses

*   **Why is there a delay on first query?**
    The orchestrator initializes the FAISS index on startup. The first query after a cold start may take 2-3 seconds longer as connections to Redis and MongoDB are established.

*   **The `[KAFKA MOCK]` line**
    You will see terminal logs indicating Kafka events. This simulates publishing to the AkadVerse Event Bus for the Insight Engine to track student engagement.

## Generated Files

| File / Folder | What it is | Gitignore? |
| :------------ | :--------- | :--------- |
| `faiss_platform_index/` | Local vector store files | Yes |
| `.env` | Sensitive API keys | Yes -- NEVER COMMIT |
| `__pycache__/` | Python byte code | Yes |

## Common Errors and Fixes

*   **Error: 429 Resource Exhausted**
    *   Cause: You have hit the Gemini Free Tier rate limit.
    *   Fix: The agent has a built-in 15-minute backoff. Wait for the cooldown or upgrade to a paid tier.

*   **Error: ConnectionRefusedError (Redis)**
    *   Cause: Redis server is not running.
    *   Fix: Ensure Redis is started (`redis-server`) or check the `REDIS_AVAILABLE` logs.

## Project Structure

```
akadverse-orchestrator/
|-- agent.py                 # Core Orchestrator logic
|-- requirements.txt         # Dependencies
|-- .env                     # API Keys (Local only)
|-- .gitignore               # Exclusions
|-- faiss_platform_index/    # Vector Store (Persistent)
|-- akadverse_demo.html      # Frontend Demo Console
```

## Part of the AkadVerse Platform

This microservice is Tier 1 in the AkadVerse AI architecture, operating within the Platform Core alongside:

*   Attendance AI (Port 8001)
*   Grade Upload AI (Port 8002)
*   Notes Creator (Port 8003)

The `agent.message.relayed` Kafka event published by this service is consumed by the specialized assistants in production.

---

AkadVerse AI Architecture -- v1.0