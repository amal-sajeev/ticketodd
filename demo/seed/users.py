# Seed data: Users (citizens, officers across every department, admin)

from .config import new_id, now_utc, pwd_context

# ---------------------------------------------------------------------------
# Raw user definitions
# ---------------------------------------------------------------------------
USERS = [
    # ---- Citizens (4) ----
    {"username": "citizen1", "password": "citizen123",
     "full_name": "Rajesh Kumar Swain", "email": "rajesh.swain@email.com",
     "phone": "9876543210", "role": "citizen", "department": None},

    {"username": "citizen2", "password": "citizen123",
     "full_name": "Anita Behera", "email": "anita.behera@email.com",
     "phone": "9876543211", "role": "citizen", "department": None},

    {"username": "citizen3", "password": "citizen123",
     "full_name": "Dambaru Majhi", "email": "dambaru.majhi@email.com",
     "phone": "9876543212", "role": "citizen", "department": None},

    {"username": "citizen4", "password": "citizen123",
     "full_name": "Kuni Sabar", "email": "kuni.sabar@email.com",
     "phone": "9876543213", "role": "citizen", "department": None,
     "has_face_id": True},

    # ---- Officers (7) — one per major department + senior officer ----
    {"username": "officer_bdo", "password": "officer123",
     "full_name": "Smt. Priya Pattnaik, BDO", "email": "priya.bdo@panchayat.odisha.gov.in",
     "phone": "9988776655", "role": "officer", "department": "panchayati_raj"},

    {"username": "officer_rwss", "password": "officer123",
     "full_name": "Er. Anil Panigrahi, EE-RWSS", "email": "anil.rwss@panchayat.odisha.gov.in",
     "phone": "9988776601", "role": "officer", "department": "rural_water_supply"},

    {"username": "officer_mgnregs", "password": "officer123",
     "full_name": "Sri Bikram Sahu, MGNREGS PO", "email": "bikram.mgnregs@panchayat.odisha.gov.in",
     "phone": "9988776602", "role": "officer", "department": "mgnregs"},

    {"username": "officer_housing", "password": "officer123",
     "full_name": "Smt. Lopamudra Jena, BDO Housing", "email": "lopamudra.housing@panchayat.odisha.gov.in",
     "phone": "9988776603", "role": "officer", "department": "rural_housing"},

    {"username": "officer_drda", "password": "officer123",
     "full_name": "Sri Ranjit Mishra, DRDA PD", "email": "ranjit.drda@panchayat.odisha.gov.in",
     "phone": "9988776604", "role": "officer", "department": "rural_livelihoods"},

    {"username": "officer_sanitation", "password": "officer123",
     "full_name": "Smt. Sarojini Das, Block Sanitation Coord.", "email": "sarojini.sbm@panchayat.odisha.gov.in",
     "phone": "9988776605", "role": "officer", "department": "sanitation"},

    {"username": "officer_senior", "password": "officer123",
     "full_name": "Sri Debashis Swain, Sr. District Officer", "email": "debashis.senior@panchayat.odisha.gov.in",
     "phone": "9988776606", "role": "officer", "department": "infrastructure"},

    # ---- Admin (1) ----
    {"username": "admin", "password": "admin123",
     "full_name": "System Administrator", "email": "admin@panchayat.odisha.gov.in",
     "phone": None, "role": "admin", "department": None},
]

# ---------------------------------------------------------------------------
# Import function
# ---------------------------------------------------------------------------
async def import_users(db) -> dict[str, str]:
    """Insert seed users into MongoDB. Returns {username: _id} mapping."""
    print("\n  Importing seed users...")
    user_ids: dict[str, str] = {}
    for u in USERS:
        uid = new_id()
        user_doc = {
            "_id": uid,
            "username": u["username"],
            "hashed_password": pwd_context.hash(u["password"]),
            "full_name": u["full_name"],
            "email": u["email"],
            "phone": u["phone"],
            "role": u["role"],
            "department": u["department"],
            "created_at": now_utc(),
        }
        if u.get("has_face_id"):
            user_doc["has_face_id"] = True
            user_doc["face_descriptor"] = []       # placeholder
        db.users.insert_one(user_doc)
        user_ids[u["username"]] = uid
        print(f"    {u['username']:20s}  ({u['role']})")
    db.users.create_index([("username", 1)], unique=True)
    print(f"  => {len(USERS)} users created")
    return user_ids
