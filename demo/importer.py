# Panchayati Raj & Drinking Water Department — Seed Data Importer
# Populates MongoDB + Qdrant with comprehensive PR&DW demo data
#
# Usage:  python demo/importer.py      (from repo root)
#     or: python importer.py           (from demo/)

import asyncio
import sys
from pathlib import Path

from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Ensure the demo package is importable when running from repo root
_demo_dir = Path(__file__).resolve().parent
if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))

from seed.config import MONGODB_URL, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_DIM
from seed.users import import_users, USERS
from seed.knowledge import import_documentation, import_service_memory, DOCUMENTATION, SERVICE_MEMORY
from seed.schemes import import_schemes, SCHEMES
from seed.grievances import import_grievances, GRIEVANCES
from seed.extras import import_extras


async def main():
    print("=" * 64)
    print("  PR&DW Grievance Portal — Data Importer")
    print("=" * 64)

    # ------------------------------------------------------------------
    # 1. Connect MongoDB
    # ------------------------------------------------------------------
    print("\n[1/7] Connecting to MongoDB...")
    mongo_client = MongoClient(MONGODB_URL)
    db = mongo_client.grievance_system
    print(f"  Connected: {MONGODB_URL}")

    # ------------------------------------------------------------------
    # 2. Connect Qdrant
    # ------------------------------------------------------------------
    print("\n[2/7] Connecting to Qdrant...")
    if QDRANT_API_KEY:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    else:
        qdrant = QdrantClient(url=QDRANT_URL)
    print(f"  Connected: {QDRANT_URL}")

    # ------------------------------------------------------------------
    # 3. Reset all collections
    # ------------------------------------------------------------------
    print("\n[3/7] Resetting collections...")
    for coll_name in ["documentation", "service_memory", "schemes"]:
        try:
            qdrant.delete_collection(coll_name)
        except Exception:
            pass
        qdrant.create_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"  Qdrant: {coll_name}")

    for mongo_coll in ["grievances", "users", "counters",
                       "systemic_issues", "forecasts", "vouches",
                       "spam_tracking", "officer_analytics"]:
        db[mongo_coll].drop()
    print("  MongoDB: grievances, users, counters, systemic_issues,")
    print("           forecasts, vouches, spam_tracking, officer_analytics")

    # ------------------------------------------------------------------
    # 4. Seed users
    # ------------------------------------------------------------------
    print("\n[4/7] Users")
    user_ids = await import_users(db)

    # ------------------------------------------------------------------
    # 5. Seed knowledge base (documentation + service memory + schemes)
    # ------------------------------------------------------------------
    print("\n[5/7] Knowledge Base")
    n_docs = await import_documentation(qdrant)
    n_mem  = await import_service_memory(qdrant)
    n_sch  = await import_schemes(qdrant)

    # ------------------------------------------------------------------
    # 6. Seed grievances
    # ------------------------------------------------------------------
    print("\n[6/7] Grievances")
    inserted_grievances = await import_grievances(db, user_ids)

    # ------------------------------------------------------------------
    # 7. Seed extras (systemic issues, forecasts, vouches, spam, anomalies)
    # ------------------------------------------------------------------
    print("\n[7/7] Extras")
    extra_counts = await import_extras(db, inserted_grievances, user_ids)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  IMPORT COMPLETE")
    print("=" * 64)
    print(f"  Users:             {len(USERS)}")
    print(f"  Documentation:     {n_docs}")
    print(f"  Service Memory:    {n_mem}")
    print(f"  Schemes:           {n_sch}")
    print(f"  Grievances:        {len(GRIEVANCES)}")
    print(f"  Systemic Issues:   {extra_counts.get('systemic_issues', 0)}")
    print(f"  Forecasts:         {extra_counts.get('forecasts', 0)}")
    print(f"  Vouches:           {extra_counts.get('vouches', 0)}")
    print(f"  Spam Tracking:     {extra_counts.get('spam_tracking', 0)}")
    print(f"  Officer Anomalies: {extra_counts.get('officer_anomalies', 0)}")
    print()
    print("  Test credentials:")
    print("    Citizen : citizen1  / citizen123")
    print("    Officer : officer_bdo / officer123")
    print("    Admin   : admin    / admin123")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
