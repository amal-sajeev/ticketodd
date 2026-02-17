# TicketOdd — AI-Powered Grievance Redressal System

An intelligent grievance management portal built for the **Odisha Panchayati Raj & Drinking Water Department**. Citizens file complaints, AI classifies and triages them, and officers resolve them through a unified dashboard. Built with FastAPI, MongoDB, Qdrant, and OpenAI.

---

## Changelog

### v2.1 — February 2026

**UI overhaul — Material 3**

- Complete redesign from glassmorphism (Frutiger Aero) to **Google Material 3** design language
- Tonal surface/elevation system with layered containers and M3 color tokens
- Typography: **Inter** (sans-serif) + **JetBrains Mono** (code), loaded from Google Fonts
- Replaced all emoji with **Material Symbols Outlined** icons throughout every template
- AI accent gradients on key surfaces (chatbot, metrics) to convey an intelligent-system aesthetic
- Responsive KPI metric cards use flexbox single-row layout that shrinks gracefully instead of wrapping
- Styled file input components (`.file-input-wrap`) replace native browser file pickers with a dashed-border drop zone, pill-shaped button, and dynamic filename label

**Added**

- **Face login & registration** — biometric authentication via `face-api.js` (client) and `numpy` descriptor matching (backend, threshold 0.5); optional face enrollment during registration, face-login tab on login page
- **Smart Extract for schemes** — upload PDF, DOCX, or images; GPT extracts scheme name, description, eligibility, department, and application steps; scanned PDFs handled via Vision API (`gpt-4o`); non-scheme documents are rejected with guidance
- **AI eligibility question generation** — auto-generate eligibility questions from scheme text; bulk backfill endpoint for all schemes missing questions
- **Scheme reference documents** — attach PDFs/images to schemes stored in GridFS; inline viewer with PDF.js and zoom controls on scheme detail pages
- **Knowledge base management UI** — full CRUD for documentation, service memory, and schemes from a single admin interface with category filters and eligibility question editor
- **SLA deadline alert system** — notification bell in officer/admin nav with badge count; dropdown shows breached, critical (<24h), and warning (<48h) grievances; polls every 60 seconds; dedicated admin panel section with countdown cards
- **Public grievance tracking** — `/track` page allows anyone to look up a grievance by tracking number without logging in; supports direct links via `?number=` query parameter; self-resolution confirmation flow works without authentication
- **Notification toast system** — `showToast(message, type)` with Material icons for success/error feedback across all pages
- **Markdown rendering** — resolutions, chatbot messages, and AI-generated content rendered via `marked` with `DOMPurify` sanitization

**New files**

- `demo/static/js/analytics.js` — analytics dashboard logic (metrics, Chart.js charts)
- `demo/static/js/chatbot.js` — chatbot frontend (message bubbles, typing indicator, grievance filing)
- `demo/seed_users.py` — standalone script to seed user accounts without grievances

**New dependencies**

- `numpy` — face descriptor Euclidean distance matching
- `opencv-python` — image processing for face recognition
- `face_recognition` — face detection and descriptor extraction
- `PyPDF2` — PDF text extraction for scheme document processing
- `python-docx` — DOCX text extraction for scheme document processing
- `Pillow` — image handling for Vision API uploads
- Frontend CDN: `face-api.js`, `marked`, `DOMPurify`, `Chart.js`, `PDF.js`

**New environment variables**

- `QDRANT_API_KEY` — optional, Qdrant authentication key
- `OPENAI_MODEL` — LLM model name (default `gpt-5-mini`)
- `OPENAI_VISION_MODEL` — vision model for scanned document extraction (default `gpt-4o`)
- `EMBEDDING_MODEL` — embedding model (default `text-embedding-3-large`)

**New endpoints**

- `POST /knowledge/schemes/process-documents` — upload and extract scheme data from documents
- `POST /knowledge/schemes/generate-questions` — AI-generate eligibility questions for a scheme
- `POST /knowledge/schemes/backfill-questions` — bulk backfill missing eligibility questions
- `PUT /knowledge/schemes/{id}/documents` — attach reference documents to a scheme
- `GET /analytics/geographical` — district-level grievance counts for the officer dashboard map

---

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
| AI | OpenAI GPT-5-mini / GPT-4o (vision) + text-embedding-3-large |
| Auth | JWT (python-jose) + bcrypt (passlib) + face-api.js biometric login |
| Frontend | Jinja2 templates + vanilla JS, Material 3 design system |
| Charts | Chart.js |
| Document Processing | PyPDF2, python-docx, Pillow, PDF.js (viewer) |
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
      css/style.css          # Material 3 global styles
      js/common.js           # Shared JS utilities, nav, API client, toasts
      js/chatbot.js          # Chatbot frontend (bubbles, typing, grievance filing)
      js/analytics.js        # Analytics dashboard (metrics, Chart.js charts)
      odisha_districts.json  # GeoJSON for district map
      odisha_census_data.json # Census data for impact scoring (30 districts)
  seed_users.py              # Standalone user seeder script
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
   QDRANT_API_KEY=...            # optional, if Qdrant requires auth
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-5-mini       # optional, default gpt-5-mini
   OPENAI_VISION_MODEL=gpt-4o    # optional, for scanned document extraction
   EMBEDDING_MODEL=text-embedding-3-large  # optional
   JWT_SECRET=your-secret-key
   OPENWEATHERMAP_API_KEY=...    # optional, for predictive forecasting
   ```

3. **Run the server**

   ```bash
   python ticketer.py
   ```

   The portal will be available at `http://localhost:8000`.

## API Overview

The system exposes 68 endpoints across these groups:

- **Auth** — register (with optional face enrollment), login (password or face), officers list, spam status, photo ID upload
- **Grievances** — CRUD, status updates, assignment, resolution, notes, deadline checks, public tracking
- **Sub-tasks** — status and assignment for cross-department dependencies
- **Public feed** — proximity-sorted public grievances, vouching with evidence
- **Systemic issues** — list, detail, status management, officer assignment
- **Forecasts** — generation, latest, historical
- **Knowledge** — documentation CRUD, service memory CRUD, schemes CRUD, smart document extraction, AI eligibility question generation, reference document management
- **Analytics** — system-wide metrics, geo-analytics, district-level geographical data
- **Admin** — user management, spam review, officer anomaly flags, SLA deadline alerts
- **Chat** — AI chatbot with grievance-filing capability
- **Health** — system health check
