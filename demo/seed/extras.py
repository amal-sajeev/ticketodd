# Seed data: Systemic issues, forecasts, vouches, spam tracking, officer anomalies

from datetime import timedelta

from .config import new_id, now_utc, geojson_point

# ---------------------------------------------------------------------------
# Helper — pick grievance IDs whose district + department match a filter
# ---------------------------------------------------------------------------
def _grv_ids(grievances: list[dict], *, district: str | None = None,
             department: str | None = None, limit: int = 5) -> list[str]:
    out = []
    for g in grievances:
        if district and g.get("district") != district:
            continue
        if department and g.get("department") != department:
            continue
        out.append(g["_id"])
        if len(out) >= limit:
            break
    return out


# ===================================================================
# 1.  SYSTEMIC ISSUES  (4)
# ===================================================================
def _build_systemic_issues(grievances: list[dict]) -> list[dict]:
    now = now_utc()
    return [
        # --- detected (3 — these show on the officer dashboard) ---
        {"_id": new_id(),
         "title": "Widespread JJM pipeline failures across Rayagada block",
         "root_cause_analysis": (
             "Analysis of 5 related grievances in Rayagada reveals a systemic contractor "
             "quality failure. The contractor for JJM pipeline laying (Contract #RGD-JJM-2025-014) "
             "used sub-standard PVC fittings and inadequate jointing material. 70% of payment "
             "was released before quality inspection. Affected habitations: 12 villages, ~5,000 "
             "families. Root cause: weak quality-assurance gate in the payment release process."
         ),
         "department": "rural_water_supply",
         "district": "Rayagada",
         "affected_area_center": geojson_point("Rayagada"),
         "affected_radius_km": 15.0,
         "grievance_ids": _grv_ids(grievances, district="Rayagada", limit=3),
         "estimated_population_affected": 22000,
         "priority": "urgent",
         "status": "detected",
         "created_at": now - timedelta(days=3),
         "updated_at": now - timedelta(days=3),
         "assigned_officer": None},

        {"_id": new_id(),
         "title": "Recurring MGNREGS wage payment delays in western Odisha blocks",
         "root_cause_analysis": (
             "Cluster analysis of 8 grievances across Kalahandi, Nuapada, and Bargarh shows "
             "a pattern of 30-60 day wage delays. Primary cause: incorrect bank account details "
             "due to manual data entry at GP level without Aadhaar-based verification. Secondary "
             "cause: FTO processing backlog at State MGNREGS cell due to staff shortage."
         ),
         "department": "mgnregs",
         "district": "Kalahandi",
         "affected_area_center": geojson_point("Kalahandi"),
         "affected_radius_km": 50.0,
         "grievance_ids": _grv_ids(grievances, department="mgnregs", limit=4),
         "estimated_population_affected": 15000,
         "priority": "high",
         "status": "detected",
         "created_at": now - timedelta(days=2),
         "updated_at": now - timedelta(days=2),
         "assigned_officer": None},

        {"_id": new_id(),
         "title": "ODF Plus sustainability failure in Kendrapara and Jajpur",
         "root_cause_analysis": (
             "Multiple grievances report non-functional SLWM units and ODF slippage in "
             "Kendrapara and Jajpur districts. Root cause: SLWM unit operators left due to "
             "irregular salary payments from GP funds. No backup staffing plan. 23 GPs at risk "
             "of losing ODF Plus certification. Pattern also linked to absence of GP-level "
             "monitoring committees."
         ),
         "department": "sanitation",
         "district": "Kendrapara",
         "affected_area_center": geojson_point("Kendrapara"),
         "affected_radius_km": 25.0,
         "grievance_ids": _grv_ids(grievances, department="sanitation", limit=3),
         "estimated_population_affected": 45000,
         "priority": "high",
         "status": "detected",
         "created_at": now - timedelta(days=1),
         "updated_at": now - timedelta(days=1),
         "assigned_officer": None},

        # --- acknowledged ---
        {"_id": new_id(),
         "title": "Fluoride contamination pattern in Jharsuguda and Sundargarh bore wells",
         "root_cause_analysis": (
             "4 grievances from adjacent blocks in Jharsuguda and Sundargarh report fluoride-"
             "affected drinking water causing dental fluorosis in children. Root cause: bore wells "
             "drilled into fluoride-bearing geological formations without pre-commissioning water "
             "quality testing. Basudha and JJM schemes both affected. Estimated 3,200 households "
             "consuming unsafe water."
         ),
         "department": "rural_water_supply",
         "district": "Jharsuguda",
         "affected_area_center": geojson_point("Jharsuguda"),
         "affected_radius_km": 20.0,
         "grievance_ids": _grv_ids(grievances, district="Jharsuguda", limit=2)
                        + _grv_ids(grievances, district="Sundargarh", limit=2),
         "estimated_population_affected": 14000,
         "priority": "high",
         "status": "acknowledged",
         "created_at": now - timedelta(days=10),
         "updated_at": now - timedelta(days=5),
         "assigned_officer": "Er. Anil Panigrahi, EE-RWSS"},

        # --- in_progress ---
        {"_id": new_id(),
         "title": "PMAY-G geo-tagging photo rejection causing mass installment delays",
         "root_cause_analysis": (
             "12 grievances from Ganjam and Puri districts identified a common pattern: "
             "PMAY-G second and third installments stuck because geo-tagged photos were "
             "rejected by the Awaas+ portal. Root cause: Block Technical Assistants using "
             "phones with poor GPS accuracy (>50m error). GPS calibration training and "
             "procurement of GPS-enabled cameras approved for all BTAs."
         ),
         "department": "rural_housing",
         "district": "Ganjam",
         "affected_area_center": geojson_point("Ganjam"),
         "affected_radius_km": 30.0,
         "grievance_ids": _grv_ids(grievances, department="rural_housing", limit=3),
         "estimated_population_affected": 8000,
         "priority": "medium",
         "status": "in_progress",
         "created_at": now - timedelta(days=25),
         "updated_at": now - timedelta(days=3),
         "assigned_officer": "Smt. Lopamudra Jena, BDO Housing"},

        # --- resolved ---
        {"_id": new_id(),
         "title": "BGBO contractor cartel inflating estimates in Sundargarh tribal blocks",
         "root_cause_analysis": (
             "Audit of 6 BGBO infrastructure complaints in Sundargarh revealed that 3 "
             "contractors — all from the same business group — were awarded 80% of projects "
             "in ITDA blocks. Estimates were inflated 15-20% above Schedule of Rates. Quality "
             "parameters compromised. Resolved: Contracts terminated, blacklisted, re-tendered "
             "with mandatory third-party quality audit."
         ),
         "department": "infrastructure",
         "district": "Sundargarh",
         "affected_area_center": geojson_point("Sundargarh"),
         "affected_radius_km": 35.0,
         "grievance_ids": _grv_ids(grievances, district="Sundargarh", limit=2),
         "estimated_population_affected": 30000,
         "priority": "high",
         "status": "resolved",
         "created_at": now - timedelta(days=45),
         "updated_at": now - timedelta(days=5),
         "assigned_officer": "Sri Debashis Swain, Sr. District Officer"},
    ]


# ===================================================================
# 2.  FORECASTS  (2)
# ===================================================================
def _build_forecasts() -> list[dict]:
    now = now_utc()
    return [
        {"_id": new_id(),
         "forecast_date": now - timedelta(days=1),
         "forecast_period_start": now,
         "forecast_period_end": now + timedelta(weeks=2),
         "predictions": [
             {"district": "Koraput", "department": "rural_water_supply",
              "baseline_count": 8, "predicted_count": 20, "spike_factor": 2.5,
              "confidence": 0.78,
              "triggers": ["heavy_rainfall", "seasonal_baseline"],
              "recommended_actions": [
                  "Pre-position mobile water purification units in flood-prone blocks",
                  "Alert RWSS field engineers for emergency pipeline repair readiness"]},
             {"district": "Rayagada", "department": "infrastructure",
              "baseline_count": 5, "predicted_count": 12, "spike_factor": 2.4,
              "confidence": 0.72,
              "triggers": ["heavy_rainfall", "monsoon_road_damage"],
              "recommended_actions": [
                  "Stockpile road repair material (gravel, bitumen) at block depots",
                  "Prepare emergency culvert kits for deployment"]},
             {"district": "Ganjam", "department": "rural_housing",
              "baseline_count": 6, "predicted_count": 9, "spike_factor": 1.5,
              "confidence": 0.60,
              "triggers": ["seasonal_baseline"],
              "recommended_actions": [
                  "Expedite pending PMAY-G geo-tag verifications before monsoon"]},
             {"district": "Kalahandi", "department": "mgnregs",
              "baseline_count": 10, "predicted_count": 15, "spike_factor": 1.5,
              "confidence": 0.65,
              "triggers": ["seasonal_baseline", "lean_agriculture_season"],
              "recommended_actions": [
                  "Open additional MGNREGS worksites for drainage and land development",
                  "Ensure FTO processing backlog is cleared before demand spike"]},
             {"district": "Mayurbhanj", "department": "rural_water_supply",
              "baseline_count": 7, "predicted_count": 14, "spike_factor": 2.0,
              "confidence": 0.70,
              "triggers": ["heat_wave", "seasonal_baseline"],
              "recommended_actions": [
                  "Deploy mobile tankers to tribal habitations with single-source dependency",
                  "Fast-track pending Basudha bore well installations"]},
         ],
         "weather_data_used": {"Koraput": True, "Rayagada": True, "Mayurbhanj": True},
         "generated_by": "system"},

        {"_id": new_id(),
         "forecast_date": now - timedelta(days=15),
         "forecast_period_start": now - timedelta(days=14),
         "forecast_period_end": now,
         "predictions": [
             {"district": "Khordha", "department": "sanitation",
              "baseline_count": 4, "predicted_count": 6, "spike_factor": 1.5,
              "confidence": 0.55,
              "triggers": ["pre_monsoon_season"],
              "recommended_actions": [
                  "Activate GP-level waste collection systems before rains"]},
             {"district": "Cuttack", "department": "infrastructure",
              "baseline_count": 6, "predicted_count": 8, "spike_factor": 1.3,
              "confidence": 0.50,
              "triggers": ["seasonal_baseline"],
              "recommended_actions": [
                  "Complete pending road pothole repairs in Cuttack district"]},
         ],
         "weather_data_used": {},
         "generated_by": "system"},
    ]


# ===================================================================
# 3.  VOUCHES  (for public grievances)
# ===================================================================

# Map citizen usernames to full names for vouch attribution
_CITIZEN_NAMES = {
    "citizen1": "Rajesh Kumar Swain",
    "citizen2": "Anita Behera",
    "citizen3": "Dambaru Majhi",
    "citizen4": "Kuni Sabar",
}

_vouch_comments = [
    "I can confirm this issue. Our village faces the same problem. Water supply has been erratic for months.",
    "This is a genuine complaint. I live nearby and have witnessed the situation firsthand.",
    "Verified — the SBM waste management system has not been working in our area either.",
    "I support this grievance. Multiple families in our ward are affected by the same issue.",
    "This is accurate. The road has been in terrible condition since last monsoon. Many accidents have occurred.",
    "Confirmed. The contractor left months ago. We need urgent intervention.",
    "Same problem in our hamlet. The hand pump has been broken for 3 months and nobody repairs it.",
    "My family is also affected. We have to walk 2 km to fetch drinking water now.",
    None,  # vouch without comment
    "I spoke to the Sarpanch about this but nothing happened. Supporting this grievance.",
    None,  # vouch without comment
    "The drainage near our school is completely blocked. Children are falling sick regularly.",
]

def _build_vouches(grievances: list[dict], user_ids: dict[str, str]) -> list[dict]:
    now = now_utc()
    public_grvs = [g for g in grievances if g.get("is_public")]
    vouches: list[dict] = []

    # Build list of (username, user_id) for citizens only
    cit_entries = [(uname, uid) for uname, uid in user_ids.items() if uname.startswith("citizen")]

    comment_idx = 0
    used_pairs: set[tuple[str, str]] = set()
    for idx, grv in enumerate(public_grvs[:6]):
        # Each public grievance gets 1-3 vouches from different citizens
        n_vouches = 3 if idx < 2 else (2 if idx < 4 else 1)
        for v_idx in range(n_vouches):
            # Find a citizen who didn't file this grievance and hasn't vouched it yet
            offset = 1
            while offset <= len(cit_entries):
                uname, uid = cit_entries[(idx + v_idx + offset) % len(cit_entries)]
                if grv.get("citizen_user_id") != uid and (grv["_id"], uid) not in used_pairs:
                    break
                offset += 1
            else:
                continue  # no eligible citizen left for this grievance
            used_pairs.add((grv["_id"], uid))
            is_anonymous = (comment_idx % 5 == 3)  # roughly 1 in 5 are anonymous
            comment = _vouch_comments[comment_idx % len(_vouch_comments)]
            comment_idx += 1
            vouches.append({
                "_id": new_id(),
                "grievance_id": grv["_id"],
                "user_id": uid,
                "user_name": _CITIZEN_NAMES.get(uname, "Citizen"),
                "is_anonymous": is_anonymous,
                "comment": comment[:500] if comment else None,
                "evidence_file_ids": [],
                "created_at": now - timedelta(days=6 - idx, hours=v_idx * 4 + idx),
            })
    return vouches


# ===================================================================
# 4.  SPAM TRACKING  (2)
# ===================================================================
def _build_spam_tracking(user_ids: dict[str, str]) -> list[dict]:
    now = now_utc()
    # Use a citizen ID that exists — citizen2 gets moderate spam, citizen4 gets blocked
    cit2 = user_ids.get("citizen2", new_id())
    cit4 = user_ids.get("citizen4", new_id())
    return [
        # Blocked user — high spam score, photo ID pending review
        {"_id": cit4,
         "filing_timestamps": [now - timedelta(hours=h) for h in range(8)],
         "duplicate_hashes": [
             {"hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2", "ts": now - timedelta(hours=2)},
             {"hash": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2", "ts": now - timedelta(hours=1)},
         ],
         "spam_score": 0.85,
         "is_blocked": True,
         "blocked_at": now - timedelta(hours=1),
         "photo_id_file_id": None,
         "photo_id_status": "pending_review",
         "admin_notified": True,
         "pattern_flags": [
             {"type": "rapid_filing", "detail": "8 filings in 1 hour", "severity": "high", "timestamp": now - timedelta(hours=1)},
             {"type": "duplicate_content", "detail": "2 identical grievances detected", "severity": "medium", "timestamp": now - timedelta(hours=1)},
         ],
         "ip_addresses": ["203.122.45.67"]},

        # Warning-level user — moderate score, not blocked
        {"_id": cit2,
         "filing_timestamps": [now - timedelta(hours=h * 4) for h in range(4)],
         "duplicate_hashes": [],
         "spam_score": 0.35,
         "is_blocked": False,
         "blocked_at": None,
         "photo_id_file_id": None,
         "photo_id_status": "none",
         "admin_notified": False,
         "pattern_flags": [
             {"type": "short_description", "detail": "Multiple filings with < 20 char descriptions", "severity": "low", "timestamp": now - timedelta(days=2)},
         ],
         "ip_addresses": ["103.45.67.89"]},
    ]


# ===================================================================
# 5.  OFFICER ANOMALIES  (2)
# ===================================================================
def _build_officer_anomalies(user_ids: dict[str, str]) -> list[dict]:
    now = now_utc()
    # Pick two officer IDs
    bdo_id    = user_ids.get("officer_bdo", new_id())
    senior_id = user_ids.get("officer_senior", new_id())
    return [
        # Rubber-stamping: fast resolutions, short text
        {"_id": bdo_id,
         "daily_resolution_counts": {
             (now - timedelta(days=d)).strftime("%Y-%m-%d"): 12 + d
             for d in range(5)
         },
         "avg_resolution_text_length": 45.0,
         "avg_time_to_resolve_hours": 0.5,
         "priority_distribution": {"low": 28, "medium": 10, "high": 2, "urgent": 0},
         "anomaly_flags": [
             {"id": new_id(), "type": "rubber_stamping",
              "detail": "Average resolution time 0.5 hours with 45-char average text length across 40 resolutions",
              "timestamp": now - timedelta(days=1), "severity": "high"},
             {"id": new_id(), "type": "generic_resolutions",
              "detail": "85% of resolutions contain identical phrasing 'Issue noted and forwarded'",
              "timestamp": now - timedelta(days=1), "severity": "medium"},
         ],
         "baseline_computed_at": now - timedelta(days=1)},

        # Cherry-picking: only low-priority cases
        {"_id": senior_id,
         "daily_resolution_counts": {
             (now - timedelta(days=d)).strftime("%Y-%m-%d"): 3 + d % 2
             for d in range(7)
         },
         "avg_resolution_text_length": 250.0,
         "avg_time_to_resolve_hours": 4.0,
         "priority_distribution": {"low": 22, "medium": 3, "high": 0, "urgent": 0},
         "anomaly_flags": [
             {"id": new_id(), "type": "cherry_picking",
              "detail": "0 urgent/high cases resolved out of 25 in 30 days — only low priority cases addressed",
              "timestamp": now - timedelta(days=2), "severity": "medium"},
         ],
         "baseline_computed_at": now - timedelta(days=2)},
    ]


# ===================================================================
# Public import function
# ===================================================================
async def import_extras(db, grievances: list[dict], user_ids: dict[str, str]) -> dict[str, int]:
    """Seed systemic issues, forecasts, vouches, spam tracking, officer anomalies."""
    counts: dict[str, int] = {}

    # --- Systemic issues ---
    print("\n  Importing systemic issues...")
    issues = _build_systemic_issues(grievances)
    for issue in issues:
        db.systemic_issues.insert_one(issue)
        # Link grievances to their systemic issue
        if issue["grievance_ids"]:
            db.grievances.update_many(
                {"_id": {"$in": issue["grievance_ids"]}},
                {"$set": {"systemic_issue_id": issue["_id"]}})
        print(f"    [{issue['status']:12s}] {issue['title'][:60]}...")
    db.systemic_issues.create_index("status")
    db.systemic_issues.create_index("department")
    counts["systemic_issues"] = len(issues)
    print(f"  => {len(issues)} systemic issues")

    # --- Forecasts ---
    print("\n  Importing forecasts...")
    forecasts = _build_forecasts()
    for fc in forecasts:
        db.forecasts.insert_one(fc)
        n_preds = len(fc["predictions"])
        print(f"    Forecast {fc['forecast_date'].strftime('%Y-%m-%d')}: {n_preds} predictions")
    db.forecasts.create_index([("forecast_date", -1)])
    counts["forecasts"] = len(forecasts)
    print(f"  => {len(forecasts)} forecasts")

    # --- Vouches ---
    print("\n  Importing vouches...")
    vouches = _build_vouches(grievances, user_ids)
    anon_count = sum(1 for v in vouches if v.get("is_anonymous"))
    for v in vouches:
        db.vouches.insert_one(v)
        label = "Anonymous" if v.get("is_anonymous") else v.get("user_name", "?")
        has_comment = "with comment" if v.get("comment") else "no comment"
        print(f"    Vouch by {label:20s} ({has_comment})")
    db.vouches.create_index("grievance_id")
    db.vouches.create_index([("grievance_id", 1), ("user_id", 1)], unique=True)
    counts["vouches"] = len(vouches)
    print(f"  => {len(vouches)} vouches ({anon_count} anonymous)")

    # --- Spam tracking ---
    print("\n  Importing spam tracking records...")
    spam_records = _build_spam_tracking(user_ids)
    for sr in spam_records:
        db.spam_tracking.update_one({"_id": sr["_id"]}, {"$set": sr}, upsert=True)
    db.spam_tracking.create_index("_id")
    counts["spam_tracking"] = len(spam_records)
    print(f"  => {len(spam_records)} spam tracking records")

    # --- Officer anomalies ---
    print("\n  Importing officer anomaly records...")
    anomalies = _build_officer_anomalies(user_ids)
    for oa in anomalies:
        db.officer_analytics.update_one({"_id": oa["_id"]}, {"$set": oa}, upsert=True)
    db.officer_analytics.create_index("_id")
    counts["officer_anomalies"] = len(anomalies)
    print(f"  => {len(anomalies)} officer anomaly records")

    return counts
