# Seed data: Systemic issues, forecasts, vouches, spam tracking, officer anomalies

from datetime import timedelta

from .config import new_id, now_utc, geojson_point

# Spread extras across 6 months for charts and reports
SIX_MONTHS_DAYS = 183

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
# 1.  SYSTEMIC ISSUES  (10) — dates spread across 6 months
# ===================================================================
def _build_systemic_issues(grievances: list[dict]) -> list[dict]:
    now = now_utc()
    return [
        # --- detected (4) ---
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
         "created_at": now - timedelta(days=30),
         "updated_at": now - timedelta(days=30),
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
         "created_at": now - timedelta(days=60),
         "updated_at": now - timedelta(days=60),
         "assigned_officer": None},

        {"_id": new_id(),
         "title": "GP governance failures across Boudh and Deogarh — Gram Sabha non-compliance",
         "root_cause_analysis": (
             "7 grievances across Boudh and Deogarh districts report systemic non-compliance "
             "with Gram Sabha mandate under Odisha Gram Panchayat Act. Key failures: no Gram "
             "Sabha held for 12+ months, beneficiary lists decided without public consultation, "
             "GP expenditure details not displayed on notice boards. Root cause: weak oversight "
             "by Block administration and absence of mandatory compliance audits."
         ),
         "department": "panchayati_raj",
         "district": "Boudh",
         "affected_area_center": geojson_point("Boudh"),
         "affected_radius_km": 40.0,
         "grievance_ids": _grv_ids(grievances, district="Boudh", limit=2)
                        + _grv_ids(grievances, district="Deogarh", limit=2),
         "estimated_population_affected": 35000,
         "priority": "high",
         "status": "detected",
         "created_at": now - timedelta(days=7),
         "updated_at": now - timedelta(days=7),
         "assigned_officer": None},

        # --- acknowledged (2) ---
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
         "created_at": now - timedelta(days=90),
         "updated_at": now - timedelta(days=75),
         "assigned_officer": "Er. Anil Panigrahi, EE-RWSS"},

        {"_id": new_id(),
         "title": "MGNREGS worksite safety violations across Bargarh and Nuapada blocks",
         "root_cause_analysis": (
             "5 grievances report lack of mandated worksite facilities (shade, drinking water, "
             "first-aid, creche) at MGNREGS worksites across Bargarh and Nuapada. Root cause: "
             "Mate supervisors not trained on safety requirements, no budget allocation for "
             "worksite amenities at GP level, and absence of Block-level safety inspections. "
             "MGNREGS operational guidelines mandate these facilities."
         ),
         "department": "mgnregs",
         "district": "Bargarh",
         "affected_area_center": geojson_point("Bargarh"),
         "affected_radius_km": 30.0,
         "grievance_ids": _grv_ids(grievances, district="Bargarh", limit=2)
                        + _grv_ids(grievances, district="Nuapada", limit=2),
         "estimated_population_affected": 12000,
         "priority": "medium",
         "status": "acknowledged",
         "created_at": now - timedelta(days=45),
         "updated_at": now - timedelta(days=30),
         "assigned_officer": "Sri Bikram Sahu, MGNREGS PO"},

        # --- in_progress (2) ---
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
         "created_at": now - timedelta(days=120),
         "updated_at": now - timedelta(days=10),
         "assigned_officer": "Smt. Lopamudra Jena, BDO Housing"},

        {"_id": new_id(),
         "title": "SHG revolving fund delays in Gajapati and Nabarangpur — bank coordination failures",
         "root_cause_analysis": (
             "6 grievances from SHGs in Gajapati and Nabarangpur report 6-12 month delays "
             "in receiving revolving funds and bank linkage loans. Root cause: banks not following "
             "RBI-NRLM lending guidelines, informal caps on SHG lending at branch level, and "
             "incomplete grading documentation by Block OLM Coordinators. Systemic coordination "
             "gap between OLM offices and banking sector."
         ),
         "department": "rural_livelihoods",
         "district": "Gajapati",
         "affected_area_center": geojson_point("Gajapati"),
         "affected_radius_km": 35.0,
         "grievance_ids": _grv_ids(grievances, department="rural_livelihoods", limit=4),
         "estimated_population_affected": 5000,
         "priority": "medium",
         "status": "in_progress",
         "created_at": now - timedelta(days=100),
         "updated_at": now - timedelta(days=15),
         "assigned_officer": "Sri Ranjit Mishra, DRDA PD"},

        # --- resolved (2) ---
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
         "created_at": now - timedelta(days=160),
         "updated_at": now - timedelta(days=130),
         "assigned_officer": "Sri Debashis Swain, Sr. District Officer"},

        {"_id": new_id(),
         "title": "ODF slippage pattern in Subarnapur — false certification corrected",
         "root_cause_analysis": (
             "3 grievances from Subarnapur exposed a pattern of false ODF declarations by "
             "GPs. Physical verification found 25-30% households without functional toilets "
             "in 8 GPs that were declared ODF. Root cause: superficial third-party verification "
             "and pressure to meet state-level targets. Resolved: ODF status revoked for 8 GPs, "
             "fresh construction drives launched, verification protocol strengthened with "
             "mandatory 100% household coverage."
         ),
         "department": "sanitation",
         "district": "Subarnapur",
         "affected_area_center": geojson_point("Subarnapur"),
         "affected_radius_km": 20.0,
         "grievance_ids": _grv_ids(grievances, district="Subarnapur", limit=2),
         "estimated_population_affected": 18000,
         "priority": "medium",
         "status": "resolved",
         "created_at": now - timedelta(days=150),
         "updated_at": now - timedelta(days=110),
         "assigned_officer": "Smt. Sarojini Das, Block Sanitation Coord."},
    ]


# ===================================================================
# 2.  FORECASTS  (4) — spread across 6 months
# ===================================================================
def _build_forecasts() -> list[dict]:
    now = now_utc()
    return [
        # Most recent (yesterday)
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

        # 15 days ago (recent past)
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

        # ~30 days ago
        {"_id": new_id(),
         "forecast_date": now - timedelta(days=32),
         "forecast_period_start": now - timedelta(days=31),
         "forecast_period_end": now - timedelta(days=17),
         "predictions": [
             {"district": "Balangir", "department": "mgnregs",
              "baseline_count": 7, "predicted_count": 11, "spike_factor": 1.6,
              "confidence": 0.62,
              "triggers": ["seasonal_baseline"],
              "recommended_actions": ["Clear FTO backlog before peak demand"]},
             {"district": "Jajpur", "department": "sanitation",
              "baseline_count": 3, "predicted_count": 5, "spike_factor": 1.7,
              "confidence": 0.58,
              "triggers": ["post_monsoon_cleanup"],
              "recommended_actions": ["Inspect SLWM units in coastal blocks"]},
         ],
         "weather_data_used": {},
         "generated_by": "system"},

        # ~90 days ago (quarterly)
        {"_id": new_id(),
         "forecast_date": now - timedelta(days=92),
         "forecast_period_start": now - timedelta(days=91),
         "forecast_period_end": now - timedelta(days=61),
         "predictions": [
             {"district": "Sundargarh", "department": "infrastructure",
              "baseline_count": 9, "predicted_count": 14, "spike_factor": 1.6,
              "confidence": 0.68,
              "triggers": ["monsoon_road_damage", "seasonal_baseline"],
              "recommended_actions": [
                  "Pre-position BGBO repair material at block depots",
                  "Activate emergency culvert repair roster"]},
             {"district": "Kendrapara", "department": "rural_water_supply",
              "baseline_count": 5, "predicted_count": 9, "spike_factor": 1.8,
              "confidence": 0.65,
              "triggers": ["heat_wave", "coastal_saline_issues"],
              "recommended_actions": ["Check JJM pipeline pressure in low-lying areas"]},
         ],
         "weather_data_used": {"Sundargarh": True, "Kendrapara": True},
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
    "citizen5": "Nilambar Sethi",
    "citizen6": "Suchitra Panda",
    "citizen7": "Trinath Barik",
    "ram_kumar": "Ram Kumar",
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
    "Our ward also has the same issue. We need collective action from the GP.",
    "I have photographs as evidence. The situation is worse than described here.",
    "This contractor is notorious for substandard work. Multiple villages have complained.",
    None,  # vouch without comment
    "We submitted a written complaint to the GP 3 months ago. No response. Supporting this.",
    "Urgent issue — my elderly mother-in-law is suffering due to this problem.",
    "Confirmed. I am a ward member and I have raised this in Panchayat meetings twice.",
    "This is long overdue. The department should have acted months ago.",
]

# Index of the public grievance to skip for 0-vouch demo (by position in public_grvs list)
_ZERO_VOUCH_INDEX = 10

def _build_vouches(grievances: list[dict], user_ids: dict[str, str],
                   evidence_ids: list[str] | None = None) -> list[dict]:
    now = now_utc()
    public_grvs = [g for g in grievances if g.get("is_public")]
    vouches: list[dict] = []
    evidence_ids = evidence_ids or []

    cit_entries = [(uname, uid) for uname, uid in user_ids.items() if uname.startswith("citizen") or uname == "ram_kumar"]

    comment_idx = 0
    vouch_serial = 0
    for idx, grv in enumerate(public_grvs[:14]):
        # Skip one public grievance to demonstrate 0-vouch state
        if idx == _ZERO_VOUCH_INDEX:
            continue

        grv_created = grv.get("created_at", now - timedelta(days=14 - idx))
        # Vary vouch counts: first 3 get 4-5, next 3 get 2-3, rest get 1-2
        if idx < 3:
            n_vouches = 5 if idx == 0 else 4
        elif idx < 6:
            n_vouches = 3 if idx < 5 else 2
        else:
            n_vouches = 2 if idx % 2 == 0 else 1

        used_uids: set[str] = set()
        for v_idx in range(n_vouches):
            offset = idx + v_idx + 1
            uname, uid = cit_entries[offset % len(cit_entries)]
            while uid == grv.get("citizen_user_id") or uid in used_uids:
                offset += 1
                uname, uid = cit_entries[offset % len(cit_entries)]
            used_uids.add(uid)
            is_anonymous = (comment_idx % 6 == 4)
            comment = _vouch_comments[comment_idx % len(_vouch_comments)]
            comment_idx += 1

            ev = []
            if evidence_ids and vouch_serial < 5:
                ev = [evidence_ids[vouch_serial % len(evidence_ids)]]

            vouch_created = grv_created + timedelta(days=1 + v_idx, hours=v_idx * 3 + idx)

            vouches.append({
                "_id": new_id(),
                "grievance_id": grv["_id"],
                "user_id": uid,
                "user_name": _CITIZEN_NAMES.get(uname, "Citizen"),
                "is_anonymous": is_anonymous,
                "comment": comment[:500] if comment else None,
                "evidence_file_ids": ev,
                "created_at": vouch_created,
            })
            vouch_serial += 1
    return vouches


# ===================================================================
# 4.  SPAM TRACKING  (5) — timestamps spread across 6 months
# ===================================================================
def _build_spam_tracking(user_ids: dict[str, str],
                         photo_id_file_id: str | None = None) -> list[dict]:
    now = now_utc()
    cit2 = user_ids.get("citizen2", new_id())
    cit4 = user_ids.get("citizen4", new_id())
    cit5 = user_ids.get("citizen5", new_id())
    cit6 = user_ids.get("citizen6", new_id())
    cit7 = user_ids.get("citizen7", new_id())
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
         "photo_id_file_id": photo_id_file_id,
         "photo_id_status": "pending_review",
         "admin_notified": True,
         "pattern_flags": [
             {"type": "rapid_filing", "detail": "8 filings in 1 hour", "severity": "high", "timestamp": now - timedelta(hours=1)},
             {"type": "duplicate_content", "detail": "2 identical grievances detected", "severity": "medium", "timestamp": now - timedelta(hours=1)},
         ],
         "ip_addresses": ["203.122.45.67"]},

        # Warning-level user — moderate score, not blocked
        {"_id": cit2,
         "filing_timestamps": [now - timedelta(days=45, hours=h * 4) for h in range(4)]
                            + [now - timedelta(days=90, hours=h * 6) for h in range(2)],
         "duplicate_hashes": [],
         "spam_score": 0.35,
         "is_blocked": False,
         "blocked_at": None,
         "photo_id_file_id": None,
         "photo_id_status": "none",
         "admin_notified": False,
         "pattern_flags": [
             {"type": "short_description", "detail": "Multiple filings with < 20 char descriptions", "severity": "low", "timestamp": now - timedelta(days=45)},
             {"type": "short_description", "detail": "Similar pattern in previous month", "severity": "low", "timestamp": now - timedelta(days=90)},
         ],
         "ip_addresses": ["103.45.67.89"]},

        # Photo ID approved, unblocked after review
        {"_id": cit5,
         "filing_timestamps": [now - timedelta(days=20, hours=h * 3) for h in range(5)],
         "duplicate_hashes": [],
         "spam_score": 0.45,
         "is_blocked": False,
         "blocked_at": None,
         "photo_id_file_id": photo_id_file_id,
         "photo_id_status": "approved",
         "admin_notified": True,
         "pattern_flags": [
             {"type": "rapid_filing", "detail": "5 filings in 15 hours", "severity": "medium", "timestamp": now - timedelta(days=20)},
         ],
         "ip_addresses": ["117.200.31.42"]},

        # Photo ID rejected, blocked
        {"_id": cit6,
         "filing_timestamps": [now - timedelta(days=10, hours=h) for h in range(12)],
         "duplicate_hashes": [
             {"hash": "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3", "ts": now - timedelta(days=10, hours=3)},
             {"hash": "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3", "ts": now - timedelta(days=10, hours=5)},
             {"hash": "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3", "ts": now - timedelta(days=10, hours=7)},
         ],
         "spam_score": 0.92,
         "is_blocked": True,
         "blocked_at": now - timedelta(days=9),
         "photo_id_file_id": photo_id_file_id,
         "photo_id_status": "rejected",
         "admin_notified": True,
         "pattern_flags": [
             {"type": "rapid_filing", "detail": "12 filings in 12 hours", "severity": "high", "timestamp": now - timedelta(days=10)},
             {"type": "duplicate_content", "detail": "3 identical grievances detected", "severity": "high", "timestamp": now - timedelta(days=10)},
             {"type": "manual_block", "detail": "Admin manually blocked after review", "severity": "high", "timestamp": now - timedelta(days=9)},
         ],
         "ip_addresses": ["182.70.45.12"]},

        # Comprehensive pattern_flags — all 9 types
        {"_id": cit7,
         "filing_timestamps": [now - timedelta(days=5, hours=h * 2) for h in range(10)],
         "duplicate_hashes": [
             {"hash": "c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", "ts": now - timedelta(days=5, hours=4)},
             {"hash": "c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", "ts": now - timedelta(days=5, hours=6)},
         ],
         "spam_score": 0.78,
         "is_blocked": True,
         "blocked_at": now - timedelta(days=4),
         "photo_id_file_id": None,
         "photo_id_status": "none",
         "admin_notified": True,
         "pattern_flags": [
             {"type": "rapid_filing", "detail": "10 filings in 20 hours", "severity": "high", "timestamp": now - timedelta(days=5)},
             {"type": "duplicate_content", "detail": "2 identical grievances by same user", "severity": "medium", "timestamp": now - timedelta(days=5)},
             {"type": "short_description", "detail": "6 filings with less than 15 character descriptions", "severity": "low", "timestamp": now - timedelta(days=5)},
             {"type": "content_similarity", "detail": "8 out of 10 filings have >85% content overlap", "severity": "high", "timestamp": now - timedelta(days=5)},
             {"type": "keyword_stuffing", "detail": "Repeated keywords: 'urgent urgent urgent help help'", "severity": "medium", "timestamp": now - timedelta(days=5)},
             {"type": "cross_user_duplicate", "detail": "Content matches filings by 2 other users from same IP", "severity": "high", "timestamp": now - timedelta(days=5)},
             {"type": "ip_correlation", "detail": "Same IP address used by 3 different user accounts", "severity": "high", "timestamp": now - timedelta(days=5)},
             {"type": "manual_block", "detail": "Admin manually blocked after pattern review", "severity": "high", "timestamp": now - timedelta(days=4)},
             {"type": "admin_flag_spam", "detail": "Admin flagged all 10 filings as spam after investigation", "severity": "high", "timestamp": now - timedelta(days=4)},
         ],
         "ip_addresses": ["203.122.45.67", "203.122.45.68"]},
    ]


# ===================================================================
# 5.  OFFICER ANOMALIES  (5) — daily counts over 6 months for charts
# ===================================================================
def _build_officer_anomalies(user_ids: dict[str, str]) -> list[dict]:
    now = now_utc()
    days_range = SIX_MONTHS_DAYS
    bdo_id      = user_ids.get("officer_bdo", new_id())
    senior_id   = user_ids.get("officer_senior", new_id())
    mgnregs_id  = user_ids.get("officer_mgnregs", new_id())
    housing_id  = user_ids.get("officer_housing", new_id())
    mgnregs2_id = user_ids.get("officer_mgnregs2", new_id())
    return [
        # Rubber-stamping: fast resolutions, short text
        {"_id": bdo_id,
         "daily_resolution_counts": {
             (now - timedelta(days=d)).strftime("%Y-%m-%d"): 5 + (d % 10)
             for d in range(days_range)
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
             (now - timedelta(days=d)).strftime("%Y-%m-%d"): 2 + (d % 4)
             for d in range(days_range)
         },
         "avg_resolution_text_length": 250.0,
         "avg_time_to_resolve_hours": 4.0,
         "priority_distribution": {"low": 22, "medium": 3, "high": 0, "urgent": 0},
         "anomaly_flags": [
             {"id": new_id(), "type": "cherry_picking",
              "detail": "0 urgent/high cases resolved out of 25 in 30 days — only low priority cases addressed",
              "timestamp": now - timedelta(days=60), "severity": "medium"},
         ],
         "baseline_computed_at": now - timedelta(days=60)},

        # Bulk resolve: 50+ resolutions in a single day
        {"_id": mgnregs_id,
         "daily_resolution_counts": {
             **(
                 {(now - timedelta(days=d)).strftime("%Y-%m-%d"): 3 + (d % 5)
                  for d in range(days_range) if d != 5}
             ),
             (now - timedelta(days=5)).strftime("%Y-%m-%d"): 54,
         },
         "avg_resolution_text_length": 120.0,
         "avg_time_to_resolve_hours": 2.0,
         "priority_distribution": {"low": 15, "medium": 30, "high": 8, "urgent": 1},
         "anomaly_flags": [
             {"id": new_id(), "type": "bulk_resolve",
              "detail": "54 grievances resolved on a single day (5 days ago) — normal daily average is 5.2",
              "timestamp": now - timedelta(days=5), "severity": "high"},
         ],
         "baseline_computed_at": now - timedelta(days=5)},

        # Generic text: copy-paste resolutions
        {"_id": housing_id,
         "daily_resolution_counts": {
             (now - timedelta(days=d)).strftime("%Y-%m-%d"): 4 + (d % 6)
             for d in range(days_range)
         },
         "avg_resolution_text_length": 38.0,
         "avg_time_to_resolve_hours": 1.5,
         "priority_distribution": {"low": 12, "medium": 18, "high": 5, "urgent": 0},
         "anomaly_flags": [
             {"id": new_id(), "type": "generic_text",
              "detail": "92% of resolutions are identical: 'Matter is being looked into. Appropriate action will be taken shortly.'",
              "timestamp": now - timedelta(days=3), "severity": "high"},
         ],
         "baseline_computed_at": now - timedelta(days=3)},

        # Compound: cherry_picking + rubber_stamping
        {"_id": mgnregs2_id,
         "daily_resolution_counts": {
             (now - timedelta(days=d)).strftime("%Y-%m-%d"): 6 + (d % 8)
             for d in range(days_range)
         },
         "avg_resolution_text_length": 55.0,
         "avg_time_to_resolve_hours": 0.8,
         "priority_distribution": {"low": 35, "medium": 8, "high": 0, "urgent": 0},
         "anomaly_flags": [
             {"id": new_id(), "type": "cherry_picking",
              "detail": "0 high/urgent cases resolved out of 43 in 30 days — 81% of resolved cases are low priority",
              "timestamp": now - timedelta(days=10), "severity": "high"},
             {"id": new_id(), "type": "rubber_stamping",
              "detail": "Average resolution time 0.8 hours with 55-char text — suggestive of perfunctory closures",
              "timestamp": now - timedelta(days=10), "severity": "medium"},
         ],
         "baseline_computed_at": now - timedelta(days=10)},
    ]


# ===================================================================
# Public import function
# ===================================================================
async def import_extras(db, grievances: list[dict], user_ids: dict[str, str],
                        *, file_ids: dict | None = None) -> dict[str, int]:
    """Seed systemic issues, forecasts, vouches, spam tracking, officer anomalies."""
    counts: dict[str, int] = {}
    file_ids = file_ids or {}

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
    vouches = _build_vouches(grievances, user_ids,
                             evidence_ids=file_ids.get("vouch_evidence"))
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
    spam_records = _build_spam_tracking(user_ids,
                                       photo_id_file_id=file_ids.get("photo_id"))
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
