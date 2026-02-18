# Panchayati Raj & Drinking Water Department — Seed Data Importer
# Populates MongoDB + Qdrant with comprehensive PR&DW demo data
#
# Usage:  python demo/importer.py      (from repo root)
#     or: python importer.py           (from demo/)

import asyncio
import sys
import urllib.request
from pathlib import Path

import gridfs
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Ensure the demo package is importable when running from repo root
_demo_dir = Path(__file__).resolve().parent
if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))

from seed.config import MONGODB_URL, QDRANT_URL, QDRANT_API_KEY, EMBEDDING_DIM, now_utc
from seed.users import import_users, USERS
from seed.knowledge import import_documentation, import_service_memory, DOCUMENTATION, SERVICE_MEMORY
from seed.schemes import import_schemes, SCHEMES
from seed.grievances import import_grievances, GRIEVANCES
from seed.extras import import_extras

_SEED_IMAGES = [
    ("IMG_5432.jpg", "image/jpeg",
     "https://huggingface.co/datasets/amal-s-turing/OSWorldData/resolve/main/IMG_5432.jpg?download=true"),
    ("big-beach.jpg", "image/jpeg",
     "https://huggingface.co/datasets/amal-s-turing/OSWorldData/resolve/main/big-beach.jpg?download=true"),
]


def _seed_gridfs_files(db) -> dict:
    """Download seed images and store them in GridFS for various attachment types.

    Returns a dict of file IDs keyed by purpose.
    """
    gfs = gridfs.GridFS(db)
    downloaded: list[tuple[str, str, bytes]] = []

    for filename, content_type, url in _SEED_IMAGES:
        print(f"    Downloading {filename}...")
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        downloaded.append((filename, content_type, data))
        print(f"      {len(data):,} bytes")

    file_ids: dict = {"grievance_attachments": [], "vouch_evidence": [], "scheme_docs": []}

    ts = now_utc().isoformat()
    for filename, content_type, data in downloaded:
        fid = gfs.put(data, filename=filename, content_type=content_type,
                      metadata={"uploaded_at": ts})
        file_ids["grievance_attachments"].append(str(fid))

    fid = gfs.put(downloaded[0][2], filename=downloaded[0][0],
                  content_type=downloaded[0][1])
    file_ids["vouch_evidence"].append(str(fid))

    fid = gfs.put(downloaded[1][2], filename="photo_id_citizen4",
                  content_type=downloaded[1][1])
    file_ids["photo_id"] = str(fid)

    for filename, content_type, data in downloaded:
        fid = gfs.put(data, filename=filename, content_type=content_type,
                      metadata={"type": "scheme_document"})
        file_ids["scheme_docs"].append({
            "id": str(fid), "filename": filename,
            "content_type": content_type, "size": len(data),
        })

    return file_ids


async def main():
    print("=" * 64)
    print("  PR&DW Grievance Portal — Data Importer")
    print("=" * 64)

    # ------------------------------------------------------------------
    # 1. Connect MongoDB
    # ------------------------------------------------------------------
    print("\n[1/8] Connecting to MongoDB...")
    mongo_client = MongoClient(MONGODB_URL)
    db = mongo_client.grievance_system
    print(f"  Connected: {MONGODB_URL}")

    # ------------------------------------------------------------------
    # 2. Connect Qdrant
    # ------------------------------------------------------------------
    print("\n[2/8] Connecting to Qdrant...")
    if QDRANT_API_KEY:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    else:
        qdrant = QdrantClient(url=QDRANT_URL)
    print(f"  Connected: {QDRANT_URL}")

    # ------------------------------------------------------------------
    # 3. Reset all collections
    # ------------------------------------------------------------------
    print("\n[3/8] Resetting collections...")
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
                       "spam_tracking", "officer_analytics",
                       "fs.files", "fs.chunks"]:
        db[mongo_coll].drop()
    print("  MongoDB: grievances, users, counters, systemic_issues,")
    print("           forecasts, vouches, spam_tracking, officer_analytics,")
    print("           fs.files, fs.chunks (GridFS)")

    # ------------------------------------------------------------------
    # 4. Seed sample files (download images → GridFS)
    # ------------------------------------------------------------------
    print("\n[4/8] Sample files (GridFS)")
    print("  Downloading seed images and storing in GridFS...")
    file_ids = _seed_gridfs_files(db)
    n_files = (len(file_ids["grievance_attachments"])
               + len(file_ids["vouch_evidence"])
               + len(file_ids["scheme_docs"])
               + (1 if file_ids.get("photo_id") else 0))
    print(f"  => {n_files} GridFS files stored")

    # ------------------------------------------------------------------
    # 5. Seed users
    # ------------------------------------------------------------------
    print("\n[5/8] Users")
    user_ids = await import_users(db)

    # ------------------------------------------------------------------
    # 6. Seed knowledge base (documentation + service memory + schemes)
    # ------------------------------------------------------------------
    print("\n[6/8] Knowledge Base")
    n_docs = await import_documentation(qdrant)
    n_mem  = await import_service_memory(qdrant)
    n_sch  = await import_schemes(qdrant)

    # Attach sample documents to first 2 schemes in Qdrant
    if file_ids["scheme_docs"]:
        _records, _offset = qdrant.scroll(collection_name="schemes", limit=20)
        _targets = ["Jal Jeevan Mission (JJM)", "PMAY-Gramin"]
        _updated = 0
        for rec in _records:
            name = rec.payload.get("name", "")
            if any(t in name for t in _targets) and _updated < len(file_ids["scheme_docs"]):
                qdrant.set_payload(
                    collection_name="schemes",
                    payload={"documents": [file_ids["scheme_docs"][_updated]]},
                    points=[rec.id],
                )
                print(f"    Attached document to scheme: {name[:50]}")
                _updated += 1
        if _updated:
            print(f"  => {_updated} scheme documents attached")

    # ------------------------------------------------------------------
    # 7. Seed grievances
    # ------------------------------------------------------------------
    print("\n[7/8] Grievances")
    inserted_grievances = await import_grievances(db, user_ids, file_ids=file_ids)

    # ------------------------------------------------------------------
    # 8. Seed extras (systemic issues, forecasts, vouches, spam, anomalies)
    # ------------------------------------------------------------------
    print("\n[8/8] Extras")
    extra_counts = await import_extras(db, inserted_grievances, user_ids, file_ids=file_ids)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  IMPORT COMPLETE")
    print("=" * 64)
    print(f"  Users:             {len(USERS)}")
    print(f"  GridFS Files:      {n_files}")
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
