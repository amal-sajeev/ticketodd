"""
FastAPI dependencies: auth, DB access, role checking.

These are imported and used with `Depends(...)` in router endpoints.
"""

from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.config import settings
from app.database import db, redis_client

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


async def get_db():
    return db


async def get_qdrant():
    from app.database import qdrant
    return qdrant


async def _is_token_revoked(token: str) -> bool:
    if redis_client:
        if await redis_client.get(f"revoked:{token}"):
            return True
    revoked = await db.revoked_tokens.find_one({"_id": token})
    return revoked is not None


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if await _is_token_revoked(token):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await db.users.find_one({"username": username})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)):
    if token is None:
        return None
    try:
        if await _is_token_revoked(token):
            return None
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            return None
        username = payload.get("sub")
        if username is None:
            return None
        return await db.users.find_one({"username": username})
    except JWTError:
        return None


def require_role(*roles):
    async def role_checker(user=Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker
