# Shared configuration, helpers, and constants for all seed modules

import os
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone
from openai import AsyncOpenAI
from passlib.context import CryptContext
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent.parent          # demo/
for _env_path in [_script_dir / ".env", _script_dir.parent / ".env", Path.cwd() / ".env"]:
    if _env_path.is_file():
        load_dotenv(_env_path, override=True)
        break
else:
    load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Connection strings
# ---------------------------------------------------------------------------
MONGODB_URL    = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM  = 3072

# ---------------------------------------------------------------------------
# Shared clients
# ---------------------------------------------------------------------------
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def new_id() -> str:
    return str(uuid.uuid4())

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

async def get_embedding(text: str) -> list[float]:
    response = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

# ---------------------------------------------------------------------------
# Officer-category mapping (department -> default officer category)
# ---------------------------------------------------------------------------
OFFICER_CATS = {
    "panchayati_raj":    "block_dev_officer",
    "rural_water_supply":"executive_engineer_rwss",
    "mgnregs":           "mgnregs_programme_officer",
    "rural_housing":     "block_dev_officer",
    "rural_livelihoods": "drda_project_director",
    "sanitation":        "block_dev_officer",
    "infrastructure":    "block_dev_officer",
    "general":           "general_officer",
}

# ---------------------------------------------------------------------------
# SLA hours by priority
# ---------------------------------------------------------------------------
PRIORITY_SLA_HOURS = {"low": 360, "medium": 168, "high": 72, "urgent": 24}

# ---------------------------------------------------------------------------
# Approximate district centre coordinates (lon, lat) for GeoJSON Points
# ---------------------------------------------------------------------------
DISTRICT_COORDS = {
    "Angul":          (85.10, 20.84),
    "Balangir":       (83.49, 20.72),
    "Balasore":       (86.93, 21.49),
    "Bargarh":        (83.62, 21.33),
    "Bhadrak":        (86.50, 21.05),
    "Boudh":          (84.32, 20.84),
    "Cuttack":        (85.88, 20.46),
    "Deogarh":        (84.73, 21.54),
    "Dhenkanal":      (85.60, 20.66),
    "Gajapati":       (84.13, 19.22),
    "Ganjam":         (84.99, 19.39),
    "Jagatsinghpur":  (86.17, 20.26),
    "Jajpur":         (86.34, 20.85),
    "Jharsuguda":     (84.01, 21.85),
    "Kalahandi":      (83.17, 19.91),
    "Kandhamal":      (84.24, 20.47),
    "Kendrapara":     (86.42, 20.50),
    "Kendujhar":      (85.58, 21.63),
    "Khordha":        (85.83, 20.18),
    "Koraput":        (82.71, 18.81),
    "Malkangiri":     (81.88, 18.35),
    "Mayurbhanj":     (86.27, 21.93),
    "Nabarangpur":    (82.55, 19.23),
    "Nayagarh":       (85.10, 20.13),
    "Nuapada":        (82.55, 20.88),
    "Puri":           (85.83, 19.81),
    "Rayagada":       (83.42, 19.17),
    "Sambalpur":      (83.97, 21.47),
    "Subarnapur":     (83.87, 20.83),
    "Sundargarh":     (84.04, 22.12),
}

def geojson_point(district: str) -> dict | None:
    """Return a GeoJSON Point for *district*, or None if unknown."""
    coords = DISTRICT_COORDS.get(district)
    if coords is None:
        return None
    return {"type": "Point", "coordinates": list(coords)}
