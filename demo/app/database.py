"""
Database client initialization and lifecycle management.

Provides async MongoDB, Redis, and Qdrant connections shared across the app.
"""

import os
from typing import Optional

import structlog
from pymongo import AsyncMongoClient, GEOSPHERE
from gridfs import AsyncGridFS
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
try:
    from redis.asyncio import Redis as AsyncRedis
except ImportError:
    AsyncRedis = None

from app.config import settings

logger = structlog.get_logger()

db_client: Optional[AsyncMongoClient] = None
db = None
gfs: Optional[AsyncGridFS] = None
qdrant: Optional[QdrantClient] = None
redis_client: Optional[AsyncRedis] = None
s3_client = None


async def init_redis():
    global redis_client
    if AsyncRedis is None:
        logger.warning("redis_skipped", reason="redis package not installed")
        return
    redis_client = AsyncRedis.from_url(settings.redis_url, decode_responses=False)
    await redis_client.ping()
    logger.info("redis_connected", url=settings.redis_url)


async def init_s3():
    global s3_client
    if settings.s3_bucket_name:
        try:
            import boto3
            s3_client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint_url or None,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            logger.info("s3_connected", bucket=settings.s3_bucket_name)
        except Exception as e:
            logger.warning("s3_init_failed", error=str(e))


async def init_mongodb():
    global db_client, db, gfs
    db_client = AsyncMongoClient(
        settings.mongodb_url,
        maxPoolSize=200,
        minPoolSize=10,
        maxIdleTimeMS=30000,
        retryWrites=True,
        retryReads=True,
        w="majority",
        wtimeoutms=5000,
        readPreference="secondaryPreferred",
    )
    db = db_client.grievance_system
    gfs = AsyncGridFS(db)

    await db.grievances.create_index("created_at")
    await db.grievances.create_index("status")
    await db.grievances.create_index("department")
    await db.grievances.create_index("priority")
    await db.grievances.create_index("tracking_number")
    await db.users.create_index([("username", 1)], unique=True)
    await db.grievances.create_index([("location", GEOSPHERE)])
    await db.grievances.create_index("citizen_user_id")
    await db.grievances.create_index("is_public")
    await db.spam_tracking.create_index("_id")
    await db.vouches.create_index("grievance_id")
    await db.systemic_issues.create_index("status")
    await db.systemic_issues.create_index("department")
    await db.officer_analytics.create_index("_id")
    await db.forecasts.create_index([("forecast_date", -1)])
    await db.revoked_tokens.create_index("expires_at", expireAfterSeconds=0)
    await db.refresh_tokens.create_index("expires_at", expireAfterSeconds=0)
    await db.refresh_tokens.create_index("user_id")

    # Vouch dedup
    pipeline = [
        {"$group": {
            "_id": {"grievance_id": "$grievance_id", "user_id": "$user_id"},
            "ids": {"$push": "$_id"}, "count": {"$sum": 1},
        }},
        {"$match": {"count": {"$gt": 1}}},
    ]
    async for doc in db.vouches.aggregate(pipeline):
        dups = doc["ids"][1:]
        if dups:
            await db.vouches.delete_many({"_id": {"$in": dups}})
    await db.vouches.create_index([("grievance_id", 1), ("user_id", 1)], unique=True)
    logger.info("db_initialized")


async def init_qdrant():
    global qdrant
    try:
        if settings.qdrant_api_key:
            qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)
        else:
            qdrant = QdrantClient(url=settings.qdrant_url)
        qdrant.get_collections()
    except Exception as e:
        logger.warning("qdrant_unavailable", error=str(e))
        qdrant = None
        return
    for collection in ["documentation", "service_memory", "schemes"]:
        try:
            names = [c.name for c in qdrant.get_collections().collections]
            if collection not in names:
                qdrant.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
                )
                logger.info("qdrant_collection_created", collection=collection)
        except Exception as e:
            logger.error("qdrant_collection_error", collection=collection, error=str(e))


async def shutdown():
    if db_client:
        db_client.close()
    if redis_client:
        await redis_client.aclose()
