"""
Load testing with Locust — targeting 1,500 concurrent users.

Run:
    locust -f tests/locustfile.py --host=http://localhost:8000

Or headless:
    locust -f tests/locustfile.py --host=http://localhost:8000 \
        --users=1500 --spawn-rate=50 --run-time=5m --headless
"""

import random
import string

from locust import HttpUser, task, between, events


class CitizenUser(HttpUser):
    """Simulates a citizen browsing, filing grievances, and using the chatbot."""

    wait_time = between(1, 5)
    weight = 7

    def on_start(self):
        username = "locust_" + "".join(random.choices(string.ascii_lowercase, k=8))
        resp = self.client.post("/auth/register", json={
            "username": username,
            "password": "loadtest123",
            "full_name": f"Load Test {username}",
            "email": f"{username}@test.local",
            "role": "citizen",
        })
        if resp.status_code == 200:
            data = resp.json()
            self.token = data["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task(3)
    def view_dashboard(self):
        self.client.get("/dashboard", headers=self.headers)

    @task(2)
    def view_schemes(self):
        self.client.get("/knowledge/schemes")

    @task(2)
    def track_grievance(self):
        self.client.get("/grievances/track/GRV-2026-000001ABCDEF", headers=self.headers)

    @task(1)
    def file_grievance(self):
        if not self.headers:
            return
        districts = ["Khordha", "Cuttack", "Puri", "Ganjam", "Balasore"]
        self.client.post("/grievances", data={
            "title": f"Load test grievance {random.randint(1, 10000)}",
            "description": "This is an automated load test grievance for performance benchmarking.",
            "language": "english",
            "district": random.choice(districts),
            "is_anonymous": "false",
            "is_public": "false",
        }, headers=self.headers)

    @task(2)
    def chat(self):
        if not self.headers:
            return
        self.client.post("/chat", json={
            "message": "What schemes are available for rural water supply?",
            "language": "english",
            "conversation_history": [],
        }, headers=self.headers)

    @task(1)
    def health_check(self):
        self.client.get("/health")


class OfficerUser(HttpUser):
    """Simulates an officer reviewing and resolving grievances."""

    wait_time = between(2, 8)
    weight = 2

    def on_start(self):
        resp = self.client.post("/auth/login", json={
            "username": "officer1",
            "password": "officer123",
        })
        if resp.status_code == 200:
            self.token = resp.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task(3)
    def view_queue(self):
        self.client.get("/grievances?limit=20", headers=self.headers)

    @task(2)
    def view_analytics(self):
        self.client.get("/analytics?days=30", headers=self.headers)

    @task(1)
    def view_systemic_issues(self):
        self.client.get("/systemic-issues", headers=self.headers)


class AdminUser(HttpUser):
    """Simulates an admin viewing reports and managing users."""

    wait_time = between(3, 10)
    weight = 1

    def on_start(self):
        resp = self.client.post("/auth/login", json={
            "username": "admin",
            "password": "admin123",
        })
        if resp.status_code == 200:
            self.token = resp.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task(3)
    def view_reports(self):
        self.client.get("/admin/reports/stats?timeframe=daily", headers=self.headers)

    @task(2)
    def view_users(self):
        self.client.get("/admin/users", headers=self.headers)

    @task(1)
    def view_spam(self):
        self.client.get("/admin/spam-flagged", headers=self.headers)

    @task(1)
    def deadline_alerts(self):
        self.client.get("/admin/deadline-alerts", headers=self.headers)
