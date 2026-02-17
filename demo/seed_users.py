"""
Seed script to create test users in local MongoDB.
Run this once to populate the users collection.
"""
from datetime import datetime, timezone
from pymongo import MongoClient
import bcrypt
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env
load_dotenv(Path(__file__).resolve().parent / ".env")

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

client = MongoClient(MONGODB_URL)
db = client.grievance_system

USERS = [
    {
        "username": "citizen1",
        "password": "citizen123",
        "full_name": "Test Citizen",
        "email": "citizen1@example.com",
        "phone": "9876543210",
        "role": "citizen",
        "department": None,
    },
    {
        "username": "officer1",
        "password": "officer123",
        "full_name": "Test Officer",
        "email": "officer1@example.com",
        "phone": "9876543211",
        "role": "officer",
        "department": "general",
    },
    {
        "username": "admin",
        "password": "admin123",
        "full_name": "System Admin",
        "email": "admin@example.com",
        "phone": "9876543212",
        "role": "admin",
        "department": None,
    },
]

def seed():
    created = 0
    skipped = 0
    for user_data in USERS:
        existing = db.users.find_one({"username": user_data["username"]})
        if existing:
            print(f"  SKIP  {user_data['username']} (already exists)")
            skipped += 1
            continue

        doc = {
            "username": user_data["username"],
            "hashed_password": hash_password(user_data["password"]),
            "full_name": user_data["full_name"],
            "email": user_data["email"],
            "phone": user_data["phone"],
            "role": user_data["role"],
            "department": user_data["department"],
            "created_at": datetime.now(timezone.utc),
        }
        db.users.insert_one(doc)
        print(f"  OK    {user_data['username']} (role: {user_data['role']})")
        created += 1

    print(f"\nDone! Created: {created}, Skipped: {skipped}")

if __name__ == "__main__":
    print(f"Connecting to: {MONGODB_URL}")
    print(f"Database: grievance_system\n")
    seed()
    client.close()
