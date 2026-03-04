"""
Celery tasks for background processing.

These tasks run in separate worker processes to avoid stealing CPU/memory
from the FastAPI request handlers.

Start the worker:
    celery -A tasks worker --loglevel=info --concurrency=4

Start the beat scheduler (periodic tasks):
    celery -A tasks beat --loglevel=info
"""

import os
import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

_script_dir = Path(__file__).resolve().parent
for _env_path in [_script_dir / ".env", _script_dir.parent / ".env", Path.cwd() / ".env"]:
    if _env_path.is_file():
        load_dotenv(_env_path, override=True)
        break
else:
    load_dotenv(override=True)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

app = Celery("ticketodd", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_soft_time_limit=300,
    task_time_limit=600,
    worker_max_tasks_per_child=200,
)

app.conf.beat_schedule = {
    "check-sla-deadlines": {
        "task": "tasks.check_sla_deadlines",
        "schedule": crontab(minute="*/15"),
    },
}

logger = logging.getLogger(__name__)


def _get_db():
    """Get a synchronous MongoDB connection for Celery workers."""
    from pymongo import MongoClient
    client = MongoClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    return client.grievance_system


def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _get_qdrant():
    from qdrant_client import QdrantClient
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    try:
        if api_key:
            client = QdrantClient(url=url, api_key=api_key, timeout=30)
        else:
            client = QdrantClient(url=url)
        client.get_collections()
        return client
    except Exception:
        return None


def _get_embedding(text: str) -> list:
    """Synchronous embedding helper for Celery workers."""
    client = _get_openai_client()
    if len(text) > 2000:
        text = text[:2000] + "..."
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        input=text,
    )
    return response.data[0].embedding


@app.task(name="tasks.pattern_analyze", bind=True, max_retries=2)
def pattern_analyze(self, user_id: str, title: str, description: str, ip: str):
    """Analyze filing patterns and update spam score."""
    try:
        db = _get_db()
        qdrant = _get_qdrant()
        text = f"{title} {description}"
        flags = []

        recent = list(
            db.grievances.find({"citizen_user_id": user_id})
            .sort("created_at", -1)
            .limit(10)
        )

        if len(recent) >= 3 and qdrant:
            try:
                new_emb = _get_embedding(text)
                high_sim_count = 0
                for g in recent[1:]:
                    old_emb = _get_embedding(f"{g['title']} {g['description']}")
                    dot = sum(a * b for a, b in zip(new_emb, old_emb))
                    mag_a = sum(a ** 2 for a in new_emb) ** 0.5
                    mag_b = sum(b ** 2 for b in old_emb) ** 0.5
                    sim = dot / (mag_a * mag_b + 1e-10)
                    if sim > 0.92:
                        high_sim_count += 1
                if high_sim_count >= 2:
                    flags.append({
                        "type": "content_similarity",
                        "detail": f"{high_sim_count} highly similar filings",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception as exc:
                logger.debug("Content-similarity check failed: %s", exc)

        words = description.split()
        if words:
            caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
            if caps_ratio > 0.5 and len(words) > 5:
                flags.append({
                    "type": "keyword_stuffing",
                    "detail": f"{caps_ratio:.0%} uppercase words",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        content_hash = hashlib.sha256(
            f"{title}|{description}".lower().strip().encode()
        ).hexdigest()
        dup_count = db.spam_tracking.count_documents({
            "_id": {"$ne": user_id},
            "duplicate_hashes.hash": content_hash,
        })
        if dup_count > 0:
            flags.append({
                "type": "cross_user_duplicate",
                "detail": f"Same content from {dup_count} other account(s)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        if ip:
            ip_users = db.spam_tracking.count_documents({
                "_id": {"$ne": user_id},
                "ip_addresses": ip,
            })
            if ip_users >= 2:
                flags.append({
                    "type": "ip_correlation",
                    "detail": f"IP shared with {ip_users} other accounts",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        if flags:
            score_add = len(flags) * 0.1
            db.spam_tracking.update_one(
                {"_id": user_id},
                {
                    "$push": {"pattern_flags": {"$each": flags, "$slice": -50}},
                    "$inc": {"spam_score": min(score_add, 0.3)},
                },
                upsert=True,
            )
    except Exception as e:
        logger.error("Pattern analysis error: %s", e)
        raise self.retry(exc=e, countdown=30)


@app.task(name="tasks.constellation_analyze", bind=True, max_retries=2)
def constellation_analyze(self, grievance_doc: dict):
    """Check if a grievance forms part of a systemic cluster."""
    try:
        db = _get_db()
        qdrant = _get_qdrant()
        if not qdrant:
            return

        location = grievance_doc.get("location")
        dept = grievance_doc.get("department")
        if not location or not dept:
            return
        coords = location.get("coordinates", [])
        if len(coords) != 2:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        nearby = list(db.grievances.find({
            "department": dept,
            "status": {"$in": ["pending", "in_progress", "escalated"]},
            "created_at": {"$gte": cutoff},
            "location": {
                "$near": {
                    "$geometry": {"type": "Point", "coordinates": coords},
                    "$maxDistance": 10000,
                }
            },
            "_id": {"$ne": grievance_doc["_id"]},
        }).limit(20))

        if len(nearby) < 4:
            return

        new_text = f"{grievance_doc['title']} {grievance_doc['description']}"
        new_emb = _get_embedding(new_text)
        similar_ids = [grievance_doc["_id"]]

        for g in nearby:
            if g.get("systemic_issue_id"):
                continue
            old_text = f"{g['title']} {g['description']}"
            old_emb = _get_embedding(old_text)
            dot = sum(a * b for a, b in zip(new_emb, old_emb))
            mag_a = sum(a ** 2 for a in new_emb) ** 0.5
            mag_b = sum(b ** 2 for b in old_emb) ** 0.5
            sim = dot / (mag_a * mag_b + 1e-10)
            if sim > 0.75:
                similar_ids.append(g["_id"])

        if len(similar_ids) < 5:
            return

        cluster_grievances = [grievance_doc] + [
            g for g in nearby if g["_id"] in similar_ids
        ]
        summaries = [
            f"- {g['title']}: {g['description'][:200]}"
            for g in cluster_grievances[:10]
        ]

        openai_client = _get_openai_client()
        prompt = (
            "Analyze these clustered grievances from the same area and department. "
            "They appear to share a common root cause.\n\n"
            + "\n".join(summaries)
            + "\n\nRespond in JSON: {\"title\": \"brief systemic issue title\", "
            "\"root_cause_analysis\": \"2-3 sentence analysis\", "
            "\"estimated_population_affected\": number}"
        )
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        import json
        analysis = json.loads(resp.choices[0].message.content)

        all_coords = [
            g.get("location", {}).get("coordinates", [0, 0])
            for g in cluster_grievances if g.get("location")
        ]
        if all_coords:
            avg_lon = sum(c[0] for c in all_coords) / len(all_coords)
            avg_lat = sum(c[1] for c in all_coords) / len(all_coords)
        else:
            avg_lon, avg_lat = coords[0], coords[1]

        pri_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        highest_pri = min(
            cluster_grievances,
            key=lambda g: pri_order.get(g.get("priority", "medium"), 2),
        )

        issue_doc = {
            "_id": str(uuid.uuid4()),
            "title": analysis.get("title", f"Systemic {dept} issue"),
            "root_cause_analysis": analysis.get("root_cause_analysis", "Multiple related complaints detected."),
            "department": dept,
            "district": grievance_doc.get("district"),
            "affected_area_center": {"type": "Point", "coordinates": [avg_lon, avg_lat]},
            "affected_radius_km": 10.0,
            "grievance_ids": similar_ids,
            "estimated_population_affected": analysis.get("estimated_population_affected", 0),
            "priority": highest_pri.get("priority", "high"),
            "status": "detected",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "assigned_officer": None,
        }
        db.systemic_issues.insert_one(issue_doc)
        db.grievances.update_many(
            {"_id": {"$in": similar_ids}},
            {"$set": {"systemic_issue_id": issue_doc["_id"]}},
        )
        logger.info("Systemic issue detected: %s (%d grievances)", issue_doc["title"], len(similar_ids))
    except Exception as e:
        logger.error("Constellation engine error: %s", e)
        raise self.retry(exc=e, countdown=60)


@app.task(name="tasks.add_to_service_memory", bind=True, max_retries=2)
def add_to_service_memory_task(self, grievance: dict, resolution: str, officer: str):
    """Add a resolved grievance to the Qdrant service memory collection."""
    try:
        qdrant = _get_qdrant()
        if not qdrant:
            return
        from qdrant_client.models import PointStruct

        query = f"{grievance['title']} {grievance['description']}"
        embedding = _get_embedding(query)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "query": query,
                "resolution": resolution,
                "category": grievance.get("department", "general"),
                "agent_name": officer,
                "tracking_number": grievance.get("tracking_number"),
                "grievance_id": str(grievance.get("_id", "")),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        qdrant.upsert(collection_name="service_memory", points=[point], wait=True)
    except Exception as e:
        logger.error("Service memory error: %s", e)
        raise self.retry(exc=e, countdown=30)


@app.task(name="tasks.officer_anomaly_check", bind=True, max_retries=2)
def officer_anomaly_check(self, officer_id: str, officer_name: str, resolution_text: str):
    """Check for anomalous officer resolution patterns."""
    try:
        db = _get_db()
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        record = db.officer_analytics.find_one({"_id": officer_id})
        if not record:
            record = {
                "_id": officer_id,
                "daily_resolution_counts": {},
                "avg_resolution_text_length": 0.0,
                "avg_time_to_resolve_hours": 0.0,
                "priority_distribution": {},
                "anomaly_flags": [],
                "baseline_computed_at": now,
            }

        daily = record.get("daily_resolution_counts", {})
        daily[today_str] = daily.get(today_str, 0) + 1
        cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        daily = {k: v for k, v in daily.items() if k >= cutoff}

        new_flags = []
        today_count = daily.get(today_str, 0)
        if today_count > 15 and len(resolution_text or "") < 50:
            new_flags.append({
                "id": str(uuid.uuid4()),
                "type": "bulk_resolve",
                "detail": f"{today_count} resolutions today with avg text <50 chars",
                "timestamp": now.isoformat(),
                "severity": "high",
            })

        if resolution_text and len(resolution_text.strip()) < 30:
            new_flags.append({
                "id": str(uuid.uuid4()),
                "type": "generic_text",
                "detail": f"Resolution text only {len(resolution_text.strip())} chars",
                "timestamp": now.isoformat(),
                "severity": "medium",
            })

        resolved_by_officer = list(db.grievances.find({
            "assigned_officer": officer_name,
            "status": "resolved",
            "created_at": {"$gte": now - timedelta(days=30)},
        }))
        pri_dist = {}
        for g in resolved_by_officer:
            p = g.get("priority", "medium")
            pri_dist[p] = pri_dist.get(p, 0) + 1
        urgent_high = pri_dist.get("urgent", 0) + pri_dist.get("high", 0)
        total_resolved = sum(pri_dist.values())
        if total_resolved > 10 and urgent_high == 0:
            new_flags.append({
                "id": str(uuid.uuid4()),
                "type": "cherry_picking",
                "detail": f"0 urgent/high cases resolved out of {total_resolved} in 30 days",
                "timestamp": now.isoformat(),
                "severity": "medium",
            })

        existing_flags = record.get("anomaly_flags", [])[-20:]
        existing_flags.extend(new_flags)

        db.officer_analytics.update_one(
            {"_id": officer_id},
            {"$set": {
                "daily_resolution_counts": daily,
                "priority_distribution": pri_dist,
                "anomaly_flags": existing_flags[-30:],
                "baseline_computed_at": now,
            }},
            upsert=True,
        )
    except Exception as e:
        logger.error("Officer anomaly detection error: %s", e)
        raise self.retry(exc=e, countdown=30)


@app.task(name="tasks.check_sla_deadlines")
def check_sla_deadlines():
    """Periodic task: auto-escalate grievances past SLA deadline."""
    db = _get_db()
    now = datetime.now(timezone.utc)
    breached = list(db.grievances.find({
        "status": {"$in": ["pending", "in_progress"]},
        "$or": [
            {"sla_deadline": {"$lt": now}},
            {"estimated_resolution_deadline": {"$lt": now, "$ne": None}},
        ],
    }).limit(100))

    escalated_count = 0
    for g in breached:
        if g["status"] != "escalated":
            note = {
                "id": str(uuid.uuid4()),
                "content": "Auto-escalated: deadline breached",
                "officer": "System",
                "note_type": "internal",
                "created_at": now,
            }
            update_fields = {"status": "escalated", "updated_at": now}
            if not g.get("assigned_officer"):
                officer = db.users.find_one({
                    "role": "officer",
                    "department": g.get("department"),
                })
                if officer:
                    update_fields["assigned_officer"] = officer["full_name"]
            db.grievances.update_one(
                {"_id": g["_id"]},
                {"$set": update_fields, "$push": {"notes": note}},
            )
            escalated_count += 1

    if escalated_count:
        logger.info("SLA check: escalated %d grievances", escalated_count)
