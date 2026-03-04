"""
File storage abstraction — supports GridFS (default) and S3.

If S3_BUCKET_NAME is set, files are stored in S3; otherwise GridFS is used.
"""

import io
import struct
from datetime import datetime, timezone
from typing import Optional, Tuple

import structlog
from bson import ObjectId

from app.config import settings
from app.database import gfs, s3_client, db

logger = structlog.get_logger()

MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def validate_file_magic(content: bytes, declared_type: str) -> bool:
    """Verify that file content matches declared MIME type via magic bytes."""
    if len(content) < 8:
        return False
    header = content[:8]
    for sig, mime in MAGIC_SIGNATURES.items():
        if header.startswith(sig):
            if declared_type == mime:
                return True
            if sig == b"PK\x03\x04" and declared_type.startswith("application/"):
                return True
            if sig == b"RIFF" and declared_type == "image/webp" and b"WEBP" in content[:12]:
                return True
            return False
    for prefix in ("audio/", "video/"):
        if declared_type.startswith(prefix):
            return True
    return True


async def upload_file(
    content: bytes,
    filename: str,
    content_type: str,
    metadata: Optional[dict] = None,
) -> str:
    """Upload a file to storage (S3 or GridFS). Returns file ID as string."""
    meta = metadata or {}
    meta.setdefault("uploaded_at", datetime.now(timezone.utc).isoformat())

    if s3_client and settings.s3_bucket_name:
        import uuid
        key = f"uploads/{uuid.uuid4()}/{filename}"
        s3_client.put_object(
            Bucket=settings.s3_bucket_name,
            Key=key,
            Body=content,
            ContentType=content_type,
            Metadata={k: str(v) for k, v in meta.items()},
        )
        return f"s3:{key}"

    fid = await gfs.put(content, filename=filename, content_type=content_type, metadata=meta)
    return str(fid)


async def download_file(file_id: str) -> Tuple[bytes, str, str]:
    """Download a file. Returns (content, filename, content_type)."""
    if file_id.startswith("s3:") and s3_client:
        key = file_id[3:]
        obj = s3_client.get_object(Bucket=settings.s3_bucket_name, Key=key)
        content = obj["Body"].read()
        ct = obj.get("ContentType", "application/octet-stream")
        fn = key.split("/")[-1]
        return content, fn, ct

    grid_out = await gfs.get(ObjectId(file_id))
    data = await grid_out.read()
    return data, grid_out.filename, grid_out.content_type


async def file_exists(file_id: str) -> bool:
    if file_id.startswith("s3:") and s3_client:
        try:
            key = file_id[3:]
            s3_client.head_object(Bucket=settings.s3_bucket_name, Key=key)
            return True
        except Exception:
            return False

    result = await db.fs.files.find_one({"_id": ObjectId(file_id)})
    return result is not None
