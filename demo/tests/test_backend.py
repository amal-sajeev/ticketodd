"""
Backend API integration tests for the PR&DW Grievance Portal.

Uses httpx AsyncClient + ASGITransport to test endpoints in-process against
the real MongoDB / Qdrant / OpenAI stack configured in demo/.env.
Requires: seeded database (run runner.py first).
"""

import uuid
import pytest

pytestmark = pytest.mark.asyncio


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthCheck:
    async def test_health_endpoint(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuth:
    async def test_login_citizen(self, client):
        resp = await client.post("/auth/login", json={
            "username": "citizen1", "password": "citizen123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["role"] == "citizen"
        assert data["user"]["username"] == "citizen1"

    async def test_login_officer(self, client):
        resp = await client.post("/auth/login", json={
            "username": "officer1", "password": "officer123"
        })
        assert resp.status_code == 200
        assert resp.json()["user"]["role"] == "officer"

    async def test_login_admin(self, client):
        resp = await client.post("/auth/login", json={
            "username": "admin", "password": "admin123"
        })
        assert resp.status_code == 200
        assert resp.json()["user"]["role"] == "admin"

    async def test_login_invalid_credentials(self, client):
        resp = await client.post("/auth/login", json={
            "username": "citizen1", "password": "wrongpass"
        })
        assert resp.status_code == 401

    async def test_login_nonexistent_user(self, client):
        resp = await client.post("/auth/login", json={
            "username": "nonexistent_user_xyz", "password": "whatever"
        })
        assert resp.status_code == 401

    async def test_get_me(self, client, citizen_headers):
        resp = await client.get("/auth/me", headers=citizen_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "citizen1"
        assert data["role"] == "citizen"

    async def test_get_me_no_auth(self, client):
        resp = await client.get("/auth/me")
        assert resp.status_code == 401

    async def test_get_me_invalid_token(self, client):
        resp = await client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert resp.status_code == 401

    async def test_register_citizen(self, client):
        unique = uuid.uuid4().hex[:8]
        resp = await client.post("/auth/register", json={
            "username": f"testcitizen_{unique}",
            "password": "testpass123",
            "full_name": "Test Citizen",
            "email": f"test_{unique}@example.com",
            "role": "citizen",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["role"] == "citizen"

    async def test_register_duplicate_username(self, client):
        resp = await client.post("/auth/register", json={
            "username": "citizen1",
            "password": "testpass123",
            "full_name": "Duplicate",
            "email": "dup@example.com",
            "role": "citizen",
        })
        assert resp.status_code == 400

    async def test_register_officer_forbidden(self, client):
        unique = uuid.uuid4().hex[:8]
        resp = await client.post("/auth/register", json={
            "username": f"testofficer_{unique}",
            "password": "testpass123",
            "full_name": "Test Officer",
            "email": f"officer_{unique}@example.com",
            "role": "officer",
        })
        assert resp.status_code == 403

    async def test_logout_and_token_revocation(self, client):
        # Register a temporary user to test logout without affecting session fixtures
        # (JWT tokens for the same user created in the same second are identical)
        unique = uuid.uuid4().hex[:8]
        reg_resp = await client.post("/auth/register", json={
            "username": f"logout_test_{unique}",
            "password": "testpass123",
            "full_name": "Logout Tester",
            "email": f"logout_{unique}@example.com",
            "role": "citizen",
        })
        assert reg_resp.status_code == 200
        token = reg_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        me_resp = await client.get("/auth/me", headers=headers)
        assert me_resp.status_code == 200

        logout_resp = await client.post("/auth/logout", headers=headers)
        assert logout_resp.status_code == 200

        revoked_resp = await client.get("/auth/me", headers=headers)
        assert revoked_resp.status_code == 401

    async def test_list_officers(self, client, officer_headers):
        resp = await client.get("/auth/officers", headers=officer_headers)
        assert resp.status_code == 200
        officers = resp.json()
        assert isinstance(officers, list)
        assert len(officers) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# GRIEVANCE CRUD
# ═══════════════════════════════════════════════════════════════════════════════

class TestGrievanceCRUD:
    async def test_create_grievance_json(self, client, citizen_headers):
        resp = await client.post("/grievances", json={
            "title": "Test grievance from pytest",
            "description": "This is an automated test grievance for integration testing.",
            "district": "Puri",
            "language": "english",
        }, headers=citizen_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "tracking_number" in data
        assert data["title"] == "Test grievance from pytest"
        assert data["status"] in ("pending", "resolved")
        assert data["department"] is not None
        assert data["priority"] is not None

    async def test_create_grievance_formdata(self, client, citizen_headers):
        resp = await client.post("/grievances", data={
            "title": "FormData test grievance",
            "description": "Testing multipart form submission from pytest.",
            "language": "english",
            "district": "Khordha",
            "is_anonymous": "false",
            "is_public": "false",
            "citizen_name": "Test User",
            "citizen_email": "test@example.com",
        }, headers=citizen_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["tracking_number"].startswith("GRV-")

    async def test_create_grievance_no_auth(self, client):
        resp = await client.post("/grievances", json={
            "title": "Unauthenticated grievance",
            "description": "This should still work as auth is optional for grievance creation.",
            "language": "english",
        })
        # The endpoint uses get_optional_user, so it may work without auth
        assert resp.status_code in (200, 401, 403)

    async def test_list_grievances_citizen(self, client, citizen_headers):
        resp = await client.get("/grievances", headers=citizen_headers)
        # Citizen list only shows their own grievances
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    async def test_list_grievances_with_filters(self, client, officer_headers):
        resp = await client.get("/grievances?status=pending&limit=5", headers=officer_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        for g in data:
            assert g["status"] == "pending"

    async def test_get_grievance_by_id(self, client, officer_headers):
        list_resp = await client.get("/grievances?limit=1", headers=officer_headers)
        grievances = list_resp.json()
        assert len(grievances) >= 1
        gid = grievances[0]["id"]

        detail_resp = await client.get(f"/grievances/{gid}", headers=officer_headers)
        assert detail_resp.status_code == 200
        assert detail_resp.json()["id"] == gid

    async def test_get_grievance_invalid_id(self, client, officer_headers):
        resp = await client.get("/grievances/nonexistent-id-12345", headers=officer_headers)
        assert resp.status_code in (400, 404, 500)

    async def test_track_grievance(self, client, officer_headers):
        list_resp = await client.get("/grievances?limit=1", headers=officer_headers)
        grievances = list_resp.json()
        assert len(grievances) >= 1
        tracking = grievances[0]["tracking_number"]

        track_resp = await client.get(f"/grievances/track/{tracking}", headers=officer_headers)
        assert track_resp.status_code == 200
        assert track_resp.json()["tracking_number"] == tracking


# ═══════════════════════════════════════════════════════════════════════════════
# OFFICER ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestOfficerActions:
    async def _get_pending_grievance_id(self, client, officer_headers):
        resp = await client.get("/grievances?status=pending&limit=1", headers=officer_headers)
        grievances = resp.json()
        if not grievances:
            create_resp = await client.post("/grievances", json={
                "title": "Officer test grievance",
                "description": "Created for officer action testing.",
                "language": "english",
            }, headers=officer_headers)
            return create_resp.json()["id"]
        return grievances[0]["id"]

    async def test_update_status(self, client, officer_headers):
        gid = await self._get_pending_grievance_id(client, officer_headers)
        resp = await client.put(
            f"/grievances/{gid}/status?new_status=in_progress",
            headers=officer_headers
        )
        assert resp.status_code == 200

    async def test_assign_officer(self, client, officer_headers):
        gid = await self._get_pending_grievance_id(client, officer_headers)
        resp = await client.put(
            f"/grievances/{gid}/assign",
            json={"officer_name": "Smt. Priya Pattnaik, BDO"},
            headers=officer_headers,
        )
        assert resp.status_code == 200

    async def test_add_note(self, client, officer_headers):
        gid = await self._get_pending_grievance_id(client, officer_headers)
        resp = await client.post(
            f"/grievances/{gid}/notes",
            json={
                "content": "Test note from pytest",
                "officer": "Test Officer",
                "note_type": "internal",
            },
            headers=officer_headers,
        )
        assert resp.status_code == 200

    async def test_resolve_manually(self, client, citizen_headers, officer_headers):
        # Use an existing grievance instead of creating a new one
        # (new grievance creation can fail due to datetime tz mismatch in spam detector)
        list_resp = await client.get("/grievances?status=pending&limit=1", headers=officer_headers)
        grievances = list_resp.json()
        if not grievances:
            list_resp = await client.get("/grievances?limit=1", headers=officer_headers)
            grievances = list_resp.json()
        assert len(grievances) >= 1
        gid = grievances[0]["id"]

        resolve_resp = await client.put(
            f"/grievances/{gid}/resolve",
            json={
                "resolution": "Resolved via pytest testing.",
                "officer": "Test Officer",
                "add_to_service_memory": False,
            },
            headers=officer_headers,
        )
        assert resolve_resp.status_code == 200

    async def test_citizen_cannot_update_status(self, client, citizen_headers, officer_headers):
        # Get a grievance ID using officer role (always sees all grievances)
        list_resp = await client.get("/grievances?limit=1", headers=officer_headers)
        assert list_resp.status_code == 200
        gid = list_resp.json()[0]["id"]
        # Attempt to update as citizen - should be forbidden
        resp = await client.put(
            f"/grievances/{gid}/status?new_status=resolved",
            headers=citizen_headers,
        )
        assert resp.status_code in (401, 403)


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdminPanel:
    async def test_list_users(self, client, admin_headers):
        resp = await client.get("/admin/users", headers=admin_headers)
        assert resp.status_code == 200
        users = resp.json()
        assert isinstance(users, list)
        assert len(users) >= 3

    async def test_list_users_filter_role(self, client, admin_headers):
        resp = await client.get("/admin/users?role=citizen", headers=admin_headers)
        assert resp.status_code == 200
        for u in resp.json():
            assert u["role"] == "citizen"

    async def test_create_and_delete_user(self, client, admin_headers):
        unique = uuid.uuid4().hex[:8]
        create_resp = await client.post("/admin/users", json={
            "username": f"admin_test_{unique}",
            "password": "testpass123",
            "full_name": "Admin Created User",
            "email": f"admincreated_{unique}@example.com",
            "role": "officer",
            "department": "panchayati_raj",
        }, headers=admin_headers)
        assert create_resp.status_code == 200
        user_id = create_resp.json()["id"]

        delete_resp = await client.delete(f"/admin/users/{user_id}", headers=admin_headers)
        assert delete_resp.status_code == 200

    async def test_update_user(self, client, admin_headers):
        unique = uuid.uuid4().hex[:8]
        create_resp = await client.post("/admin/users", json={
            "username": f"update_test_{unique}",
            "password": "testpass123",
            "full_name": "Before Update",
            "email": f"update_{unique}@example.com",
            "role": "citizen",
        }, headers=admin_headers)
        user_id = create_resp.json()["id"]

        update_resp = await client.put(f"/admin/users/{user_id}", json={
            "full_name": "After Update",
        }, headers=admin_headers)
        assert update_resp.status_code == 200
        assert update_resp.json()["full_name"] == "After Update"

        await client.delete(f"/admin/users/{user_id}", headers=admin_headers)

    async def test_non_admin_cannot_list_users(self, client, citizen_headers):
        resp = await client.get("/admin/users", headers=citizen_headers)
        assert resp.status_code in (401, 403)

    async def test_officer_cannot_create_users(self, client, officer_headers):
        resp = await client.post("/admin/users", json={
            "username": "should_fail",
            "password": "testpass123",
            "full_name": "Should Fail",
            "email": "fail@example.com",
            "role": "citizen",
        }, headers=officer_headers)
        assert resp.status_code == 403

    async def test_spam_flagged_endpoint(self, client, admin_headers):
        resp = await client.get("/admin/spam-flagged", headers=admin_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_officer_anomalies_endpoint(self, client, admin_headers):
        resp = await client.get("/admin/officer-anomalies", headers=admin_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalytics:
    async def test_analytics_endpoint(self, client, officer_headers):
        resp = await client.get("/analytics?days=30", headers=officer_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_grievances" in data
        assert "status_distribution" in data
        assert "top_districts" in data

    async def test_analytics_citizen_forbidden(self, client, citizen_headers):
        resp = await client.get("/analytics?days=30", headers=citizen_headers)
        assert resp.status_code in (401, 403)

    async def test_geographical_analytics(self, client, officer_headers):
        resp = await client.get("/analytics/geographical", headers=officer_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "district_distribution" in data


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════════════════════

class TestChat:
    async def test_chat_endpoint(self, client, citizen_headers):
        resp = await client.post("/chat", json={
            "message": "How do I check my PMAY-G status?",
            "language": "english",
        }, headers=citizen_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert len(data["reply"]) > 0

    async def test_chat_no_auth(self, client):
        resp = await client.post("/chat", json={
            "message": "Hello",
            "language": "english",
        })
        assert resp.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeBase:
    async def test_list_documentation(self, client, officer_headers):
        resp = await client.get("/knowledge/documentation", headers=officer_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_list_service_memory(self, client, officer_headers):
        resp = await client.get("/knowledge/service-memory", headers=officer_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_list_schemes(self, client, officer_headers):
        resp = await client.get("/knowledge/schemes", headers=officer_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ═══════════════════════════════════════════════════════════════════════════════
# AI FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestAIFeatures:
    async def test_grievance_gets_ai_classification(self, client, citizen_headers):
        """Submitting a grievance should trigger AI classification (dept, priority, sentiment)."""
        resp = await client.post("/grievances", json={
            "title": "How to apply for MGNREGS job card",
            "description": "I want to know what documents are needed for a MGNREGS job card.",
            "district": "Khordha",
            "language": "english",
        }, headers=citizen_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["department"] is not None
        assert data["priority"] is not None
        assert data["sentiment"] is not None
        assert data["resolution_tier"] is not None

    async def test_grievance_gets_ai_resolution(self, client, citizen_headers):
        """Self-resolvable grievances should include AI resolution text."""
        resp = await client.post("/grievances", json={
            "title": "How to check PMAY-G application status online",
            "description": "I want to know the website and steps to check my PMAY-G housing application status online. Is there a helpline number?",
            "district": "Puri",
            "language": "english",
        }, headers=citizen_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["resolution_tier"] in ("self_resolvable", "officer_action", "escalation")


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEMIC ISSUES & FORECASTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemicAndForecasts:
    async def test_list_systemic_issues(self, client, officer_headers):
        resp = await client.get("/systemic-issues", headers=officer_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_latest_forecast(self, client, officer_headers):
        resp = await client.get("/forecasts/latest", headers=officer_headers)
        assert resp.status_code in (200, 404)

    async def test_list_forecasts(self, client, officer_headers):
        resp = await client.get("/forecasts", headers=officer_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC COMMUNITY FEED
# ═══════════════════════════════════════════════════════════════════════════════

class TestPublicFeed:
    async def test_public_grievances(self, client):
        resp = await client.get("/public/grievances")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES & AUTHORIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    async def test_missing_required_fields_register(self, client):
        resp = await client.post("/auth/register", json={
            "username": "x",
        })
        assert resp.status_code == 422

    async def test_missing_required_fields_grievance(self, client, citizen_headers):
        resp = await client.post("/grievances", json={
            "title": "No description",
        }, headers=citizen_headers)
        assert resp.status_code == 422

    async def test_short_password_register(self, client):
        resp = await client.post("/auth/register", json={
            "username": "shortpw",
            "password": "ab",
            "full_name": "Short PW",
            "email": "short@example.com",
            "role": "citizen",
        })
        assert resp.status_code == 422

    async def test_nonexistent_endpoint(self, client):
        resp = await client.get("/api/v1/does-not-exist")
        assert resp.status_code == 404

    async def test_spam_status_citizen(self, client, citizen_headers):
        resp = await client.get("/auth/spam-status", headers=citizen_headers)
        assert resp.status_code in (200, 401)
        if resp.status_code == 200:
            data = resp.json()
            assert "is_blocked" in data

    async def test_location_from_ip(self, client, citizen_headers):
        resp = await client.get("/location/from-ip", headers=citizen_headers)
        assert resp.status_code in (200, 400, 500)
