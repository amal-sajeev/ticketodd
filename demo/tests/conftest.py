"""
Shared pytest fixtures for the PR&DW Grievance Portal test suite.

Provides an httpx AsyncClient and pre-authenticated headers for each role.
Requires: seeded database (run runner.py first) and valid .env credentials.
"""

import sys
import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
import httpx

# Ensure the demo package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ticketer import app, lifespan, limiter


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def client(event_loop):
    """In-process httpx AsyncClient (session-scoped for performance)."""
    # Disable rate limiting during tests so login fixtures aren't throttled
    limiter.enabled = False

    async with lifespan(app):
        # Clear spam tracking for seeded test accounts so previous runs
        # don't cause 403 blocks on grievance creation
        from ticketer import db as _db
        if _db is not None:
            _db.spam_tracking.delete_many({"_id": {"$regex": "^.*$"}})

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c


async def _login(client: httpx.AsyncClient, username: str, password: str) -> dict:
    """Log in and return Authorization headers dict."""
    resp = await client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed for {username}: {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture(scope="session")
async def citizen_headers(client):
    """Auth headers for the seeded citizen1 account."""
    return await _login(client, "citizen1", "citizen123")


@pytest_asyncio.fixture(scope="session")
async def officer_headers(client):
    """Auth headers for the seeded officer1 account."""
    return await _login(client, "officer1", "officer123")


@pytest_asyncio.fixture(scope="session")
async def admin_headers(client):
    """Auth headers for the seeded admin account."""
    return await _login(client, "admin", "admin123")
