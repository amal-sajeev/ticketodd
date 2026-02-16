# TicketOdd — AI-Powered Grievance Redressal System

An intelligent grievance management portal built for the **Odisha Panchayati Raj & Drinking Water Department**. Citizens file complaints, AI classifies and triages them, and officers resolve them through a unified dashboard. Built with FastAPI, MongoDB, Qdrant, and OpenAI.

---

## Changelog

### v2.0 — February 2026

**Added**

- **File uploads** — citizens can attach up to 5 photos, audio, or video files (50MB each) per grievance, stored in MongoDB GridFS
- **Auto location detection** — cascading GPS-then-IP geolocation fills in coordinates automatically when filing
- **Spam detection** — automatic blocking of repeat/duplicate filers with photo ID verification to unblock
- **Passive pattern analysis** — background analysis of filing patterns (content similarity, keyword quality, cross-user duplicates, IP correlation)
- **Officer anomaly detection** — flags suspicious officer behavior: bulk rubber-stamping, cherry-picking easy cases, generic resolution text
- **Citizen Impact Score (0--100)** — replaces simple 4-level priority with a 5-factor score: survival necessity, district vulnerability, duration, recurrence, seasonal urgency. Uses Census 2011 data for all 30 Odisha districts.
- **Systemic issue detection ("Constellation Engine")** — clusters nearby same-department grievances using geospatial + semantic similarity, then generates a root-cause analysis via GPT
- **Predictive grievance forecasting** — generates 2-week spike predictions per district/department using historical patterns and OpenWeatherMap weather data
- **Service-memory deadlines** — estimates resolution time from similar past cases and auto-escalates on breach
- **Cross-department dependency graph** — GPT-based decomposition of multi-department grievances into linked sub-tasks with a dependency chain
- **Scheme-grievance auto-matching** — matches open grievances against the schemes knowledge base with eligibility assessment
- **Date-grouped queues** — dashboard and officer queue now group grievances by date and sort by priority within each group
- **Public community feed** — citizens can opt to make grievances public; other citizens can vouch and submit additional evidence, sorted by proximity
- **Admin panels** — new sections for reviewing spam-flagged users and officer accountability anomalies
- **Forecast generation** — admins can trigger a predictive briefing from the admin panel

**New files**

- `demo/templates/community.html` — public social grievance feed
- `demo/templates/systemic_issue_detail.html` — systemic issue detail page
- `demo/static/odisha_census_data.json` — district-level census/vulnerability data for impact scoring

**New dependencies**

- `python-multipart` — multipart form handling for file uploads
- `httpx` — async HTTP client for IP geolocation and weather APIs

**New MongoDB collections**

- `spam_tracking` — per-user spam scores, flags, photo ID status
- `vouches` — citizen support and evidence for public grievances
- `systemic_issues` — clustered issue reports with root-cause analysis
- `officer_analytics` — per-officer resolution baselines and anomaly flags
- `forecasts` — predictive briefings with weather and seasonal data

**New environment variables**

- `OPENWEATHERMAP_API_KEY` — optional, enables weather-correlated forecasting

**Removed**

- Nothing removed. All existing functionality is preserved.

---

### v1.0 — Initial Release

- Grievance filing with AI classification (department, priority, sentiment)
- Three-tier resolution: self-resolvable (AI guidance), officer-action (AI draft for review), escalation
- Service memory and knowledge base with vector search (Qdrant)
- Government schemes database with eligibility questions
- AI chatbot with grievance-filing integration
- Officer queue with filters, assignment, manual resolution, notes
- Citizen dashboard with self-resolve confirmation flow
- Admin panel for user management
- Analytics dashboard with district heatmap
- SLA deadline tracking
- JWT authentication with role-based access (citizen, officer, admin)
- Rate limiting via SlowAPI

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| Database | MongoDB (PyMongo) + GridFS for file storage |
| Vector Search | Qdrant |
| AI | OpenAI GPT-4 + text-embedding-3-large |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| Frontend | Jinja2 templates + vanilla JavaScript |
| Rate Limiting | SlowAPI |
| HTTP Client | httpx |

## Project Structure

```
ticketodd-main/
  demo/
    ticketer.py              # Main FastAPI application (all backend logic)
    requirements.txt         # Python dependencies
    runner.py                # Test/seed runner
    importer.py              # Data import utilities
    templates/
      base.html              # Base template with nav
      login.html             # Login page
      register.html          # Registration page
      dashboard.html         # Citizen dashboard (date-grouped)
      file_grievance.html    # File grievance form (uploads, auto-location)
      grievance_detail.html  # Grievance detail (attachments, deadlines, sub-tasks)
      queue.html             # Officer grievance queue (date-grouped, impact scores)
      officer_dashboard.html # Officer dashboard (systemic issues, forecasts)
      admin.html             # Admin panel (spam review, officer anomalies)
      community.html         # Public social grievance feed
      systemic_issue_detail.html  # Systemic issue detail page
      track.html             # Grievance tracking
      chatbot.html           # AI chatbot
      schemes.html           # Schemes listing
      scheme_detail.html     # Scheme detail with eligibility
      knowledge.html         # Knowledge base management
      analytics.html         # Analytics dashboard
    static/
      css/style.css          # Global styles
      js/common.js           # Shared JS utilities, nav, API client
      odisha_districts.json  # GeoJSON for district map
      odisha_census_data.json # Census data for impact scoring (30 districts)
```

## Setup

1. **Install dependencies**

   ```bash
   cd demo
   pip install -r requirements.txt
   ```

2. **Set environment variables**

   ```
   MONGODB_URL=mongodb://localhost:27017
   QDRANT_URL=http://localhost:6333
   OPENAI_API_KEY=sk-...
   JWT_SECRET=your-secret-key
   OPENWEATHERMAP_API_KEY=...   # optional, for predictive forecasting
   ```

3. **Run the server**

   ```bash
   python ticketer.py
   ```

   The portal will be available at `http://localhost:8000`.

## API Overview

The system exposes 61 endpoints across these groups:

- **Auth** — register, login, officers list, spam status, photo ID upload
- **Grievances** — CRUD, status updates, assignment, resolution, notes, deadline checks
- **Sub-tasks** — status and assignment for cross-department dependencies
- **Public feed** — proximity-sorted public grievances, vouching with evidence
- **Systemic issues** — list, detail, status management, officer assignment
- **Forecasts** — generation, latest, historical
- **Knowledge** — documentation, service memory, schemes with eligibility
- **Analytics** — system-wide metrics, geo-analytics
- **Admin** — user management, spam review, officer anomaly flags
- **Chat** — AI chatbot with grievance-filing capability
- **Health** — system health check
