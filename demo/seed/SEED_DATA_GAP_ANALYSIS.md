# Seed Data Gap Analysis — States & Scenarios Not Accounted For

This document cross-references the application’s enums, statuses, and UI against the current seed data and the earlier expansion plan. Items marked **Not in plan** were not in the original plan; **In plan but not done** were planned but not yet implemented.

---

## 1. Grievance-related

| State / scenario | App source | Current seed | Gap |
|------------------|------------|--------------|-----|
| **Grievance status** | `GrievanceStatus`: pending, in_progress, resolved, escalated | All four present | None |
| **resolution_feedback** (1–5) | `GrievanceResponse`, grievance_detail UI | Only 4 and 5 | **Missing 1, 2, 3** — no “bad” or “average” feedback for demo |
| **citizen_phone** | Optional on create; stored on grievance | Always `None` in import | **Missing** — no grievance with phone set (file_grievance / tracking) |
| **scheme_match** shape | ticketer sets: scheme_id, scheme_name, eligibility_likely, eligibility_reasoning | Seed uses: scheme_name, relevance_score, **reasoning** (no scheme_id, no eligibility_likely) | **Gap:** UI expects `eligibility_likely`, `eligibility_reasoning`, and `scheme_id` for “View scheme details” link. Seed needs: add `eligibility_likely` (bool), `eligibility_reasoning` (or alias from reasoning), and `scheme_id` (resolve from Qdrant at import or placeholder) |
| **Sub-task status** | pending, in_progress, resolved | All three used in sub_tasks | None |
| **Note types** | internal, citizen_facing | Both used | None |
| **Districts** | 30 in `DISTRICT_COORDS` | All 30 appear in grievances | None |
| **Department** | All 8 | All 8 | None |
| **Priority / Sentiment / Language / Resolution type & tier** | As in enums | Covered | None |

---

## 2. Spam tracking & photo ID

| State / scenario | App source | Current seed | Gap |
|------------------|------------|--------------|-----|
| **photo_id_status** | `none`, `pending_review`, `approved`, `rejected` (file_grievance.html, spam_flag_detail, admin review) | Only **none** and **pending_review** | **Missing approved:** user blocked → uploaded photo ID → admin approved (unblocked). **Missing rejected:** admin rejected photo ID (user stays blocked). |
| **pattern_flags types** | ticketer + admin: rapid_filing, duplicate_content, short_description, content_similarity, keyword_stuffing, cross_user_duplicate, ip_correlation, manual_block, admin_flag_spam | Only rapid_filing, duplicate_content, short_description | **Missing in seed:** content_similarity, keyword_stuffing, cross_user_duplicate, ip_correlation, **manual_block** (admin block with reason), **admin_flag_spam** (admin flag without block). Needed for spam-flag detail and admin panel demos. |
| **is_blocked + photo_id_status** | Blocked user can have pending_review / approved / rejected | One blocked + pending_review; one not blocked + none | **Missing:** blocked + rejected; unblocked after approved (optional second user). |

---

## 3. Officer analytics / anomalies

| State / scenario | App source | Current seed | Gap |
|------------------|------------|--------------|-----|
| **anomaly_flags types** | ticketer: rubber_stamping, generic_resolutions, bulk_resolve, generic_text, cherry_picking | rubber_stamping, generic_resolutions, cherry_picking | **Missing:** **bulk_resolve** (e.g. 50+ resolutions in one day), **generic_text** (copy-paste resolutions). |
| **Officer category** | OfficerCategory: block_dev_officer, district_panchayat_officer, executive_engineer_rwss, drda_project_director, gp_secretary, mgnregs_programme_officer, general_officer, senior_officer | OFFICER_CATS maps dept → category; no DPO or GP Secretary user | **Missing:** No grievance explicitly assigned to **district_panchayat_officer** or **gp_secretary** (would require an officer user in that role or a grievance with that officer_category). Optional for demo. |

---

## 4. Systemic issues

| State / scenario | App source | Current seed | Gap |
|------------------|------------|--------------|-----|
| **Systemic issue status** | detected, acknowledged, in_progress, resolved | All four present | None |

---

## 5. Users & roles

| State / scenario | App source | Current seed | Gap |
|------------------|------------|--------------|-----|
| **UserRole** | citizen, officer, admin | All three | None |
| **has_face_id** | Used for face-login / verification | citizen4 has it | None |
| **Officers per department** | Queue can filter by department | One officer per department (no second officer) | **Not in plan:** Second officer in one department for “workload distribution” / reassignment demo. Optional. |
| **Subadmin / department admin** | Plan mentioned subadmin | No subadmin user | **In plan but not done** if we still want it. |

---

## 6. Edge cases / UX scenarios

| Scenario | Purpose | Current seed | Gap |
|----------|---------|--------------|-----|
| **Citizen with zero grievances** | Empty citizen dashboard | Every citizen has at least one grievance (via citizen_key) | **Missing** — one citizen (e.g. new user) with no grievances for empty state. |
| **Public grievance with no vouches** | Contrast with highly vouched ones | All public grievances get 1–3 vouches | **Missing** — at least one is_public grievance with 0 vouches. |
| **Resolved grievance with only internal notes** | Officer timeline without citizen-facing text | Mix of internal + citizen_facing | Could add one resolved with **only internal** notes. |
| **Grievance with no AI resolution** | Pending, not yet classified | Many pending have no ai_resolution | Covered. |
| **SLA breached (pending and in_progress)** | Deadline alerts / admin | sla_breached used | Covered. |

---

## 7. Summary: what to add

**High impact (demo visibility)**  
1. **scheme_match:** Add `eligibility_likely`, `eligibility_reasoning` (or map from `reasoning`), and `scheme_id` (from Qdrant at import or consistent placeholder) so scheme match card and “View scheme details” work.  
2. **photo_id_status:** Add one spam record with **approved** (and optionally one with **rejected**).  
3. **Spam pattern_flags:** Add at least one record with **manual_block** or **admin_flag_spam**; optionally content_similarity, keyword_stuffing, cross_user_duplicate, ip_correlation.  
4. **Officer anomalies:** Add **bulk_resolve** and **generic_text** to one officer’s anomaly_flags.  
5. **resolution_feedback:** Add at least one resolved grievance with feedback 1, 2, or 3.  
6. **citizen_phone:** Set on at least one grievance (and optionally on create in runner/importer).

**Medium / optional**  
7. One **citizen with no grievances** (e.g. new user).  
8. One **is_public** grievance with **0 vouches**.  
9. One resolved grievance with **only internal** notes.  
10. **Subadmin** user and/or second officer per department (if desired).

**Already covered**  
- All grievance statuses, departments, priorities, sentiments, languages, resolution types/tiers.  
- All 30 districts.  
- Sub-task statuses, note types, systemic issue statuses.  
- Dates spread over 6 months (after your date-spread changes).

---

## 8. File-level checklist

| File | Add / change |
|------|----------------|
| **grievances.py** | resolution_feedback 1/2/3 on one or two resolved; citizen_phone on one or two; scheme_match: add eligibility_likely, eligibility_reasoning (or alias reasoning), and note scheme_id (importer). |
| **grievances import** | When building doc, if scheme_match present and has scheme_name, resolve scheme_id from Qdrant by name (or leave null if UI tolerates). Set eligibility_reasoning = scheme_match.get("eligibility_reasoning") or scheme_match.get("reasoning"). |
| **extras.py (spam)** | One record photo_id_status=approved (and optionally one rejected). Add pattern_flags: manual_block, admin_flag_spam; optionally content_similarity, keyword_stuffing, etc. |
| **extras.py (officer anomalies)** | Add anomaly_flags with type bulk_resolve and type generic_text (same or different officer). |
| **users.py** | Optional: one citizen who has no grievances (e.g. citizen9); optional subadmin. |
| **extras.py (vouches)** | Ensure at least one public grievance is not in the list that gets vouches (so it has 0 vouches), or add a separate public grievance that gets no vouches. |

This re-examination is scoped to “states and scenarios not accounted for” and can be used to extend the plan or implement the missing seed data.
