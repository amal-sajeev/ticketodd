"""
Frontend page and static asset tests for the PR&DW Grievance Portal.

Verifies that all Jinja2-rendered pages respond with 200 and valid HTML,
and that static assets (CSS, JS) are served correctly.
"""

import pytest

pytestmark = pytest.mark.asyncio

# All page routes from ticketer.py PAGES list
PAGE_ROUTES = [
    "/",
    "/login",
    "/register",
    "/dashboard",
    "/file-grievance",
    "/track",
    "/chatbot",
    "/schemes",
    "/officer-dashboard",
    "/queue",
    "/grievance-detail",
    "/knowledge",
    "/scheme-detail",
    "/analytics-view",
    "/admin",
    "/community",
    "/systemic-issue-detail",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPageRoutes:
    @pytest.mark.parametrize("route", PAGE_ROUTES)
    async def test_page_returns_200(self, client, route):
        """Each page route should return HTTP 200 with HTML content."""
        resp = await client.get(route)
        assert resp.status_code == 200, f"Route {route} returned {resp.status_code}"
        assert "text/html" in resp.headers.get("content-type", ""), (
            f"Route {route} content-type is {resp.headers.get('content-type')}"
        )

    @pytest.mark.parametrize("route", PAGE_ROUTES)
    async def test_page_contains_html_structure(self, client, route):
        """Each page should contain basic HTML structure markers."""
        resp = await client.get(route)
        text = resp.text.lower()
        assert "<!doctype html>" in text or "<html" in text, (
            f"Route {route} missing HTML doctype/tag"
        )
        assert "</html>" in text, f"Route {route} missing closing </html>"

    async def test_login_page_has_form(self, client):
        resp = await client.get("/login")
        assert "form" in resp.text.lower()

    async def test_register_page_has_form(self, client):
        resp = await client.get("/register")
        assert "form" in resp.text.lower()

    async def test_pages_have_no_cache_headers(self, client):
        """Page routes should have no-cache headers (as set in ticketer.py)."""
        resp = await client.get("/login")
        cache_control = resp.headers.get("cache-control", "")
        assert "no-cache" in cache_control or "no-store" in cache_control

    async def test_pages_have_security_headers(self, client):
        """Security middleware should set protective headers."""
        resp = await client.get("/login")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC ASSET TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStaticAssets:
    async def test_css_served(self, client):
        resp = await client.get("/static/css/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers.get("content-type", "")
        assert len(resp.text) > 100

    async def test_common_js_served(self, client):
        resp = await client.get("/static/js/common.js")
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "javascript" in ct or "text/plain" in ct
        assert len(resp.text) > 100

    async def test_districts_json_served(self, client):
        resp = await client.get("/static/odisha_census_data.json")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (dict, list))

    async def test_nonexistent_static_404(self, client):
        resp = await client.get("/static/does_not_exist.xyz")
        assert resp.status_code == 404
