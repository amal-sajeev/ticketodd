"""
Test script: Submit grievances and verify AI resolution visibility for citizens.
Requires the demo server to be running at http://localhost:8000.
"""

import requests
import json
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = "http://localhost:8000"

# Test grievances designed to be classified as self_resolvable
TEST_GRIEVANCES = [
    {
        "title": "How to check my PMAY-G housing application status",
        "description": "I applied for PMAY-G housing at my GP office in Puri 2 months ago. I don't know the website to check my status. Is there a helpline I can call?",
        "district": "Puri",
        "language": "english",
    },
    {
        "title": "What documents do I need for MGNREGS job card",
        "description": "I want to apply for MGNREGS job card in Khordha district. What papers do I need to bring? Is there a fee? How long does it take?",
        "district": "Khordha",
        "language": "english",
    },
    {
        "title": "How to get village water tested for free",
        "description": "People in our village in Dhenkanal say the borewell water has too much iron. Is there a government lab that tests water for free? What is the helpline number?",
        "district": "Dhenkanal",
        "language": "english",
    },
    {
        "title": "How to form a Self Help Group for women in our village",
        "description": "We are 12 women from poor families in Gajapati district. We want to form a group and save money together. How do we register? Who do we contact for the Rs 15000 revolving fund?",
        "district": "Gajapati",
        "language": "english",
    },
    {
        "title": "When should Gram Sabha meeting be held",
        "description": "Our Sarpanch in Boudh has not held any Gram Sabha for over a year. How many times a year should it be held? How many signatures do we need to demand a special meeting?",
        "district": "Boudh",
        "language": "english",
    },
]


def main():
    print("=" * 70)
    print("AI Resolution Test — Self-Resolvable Grievances")
    print("=" * 70)

    # Step 1: Login as citizen1
    print("\n[1] Logging in as citizen1...")
    resp = requests.post(f"{BASE}/auth/login", json={
        "username": "citizen1", "password": "citizen123"
    })
    if resp.status_code != 200:
        print(f"  FAILED to login: {resp.status_code} {resp.text}")
        sys.exit(1)
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"  Logged in. Token: {token[:20]}...")

    # Step 2: Submit each grievance and check the response
    print(f"\n[2] Submitting {len(TEST_GRIEVANCES)} test grievances...\n")

    results = []
    for i, g in enumerate(TEST_GRIEVANCES, 1):
        print(f"  [{i}/{len(TEST_GRIEVANCES)}] \"{g['title'][:50]}...\"")
        resp = requests.post(f"{BASE}/grievances", json=g, headers=headers)
        if resp.status_code != 200:
            print(f"    FAILED: {resp.status_code} {resp.text[:200]}")
            results.append({"title": g["title"], "status": "SUBMIT_FAILED"})
            continue

        data = resp.json()
        tracking = data.get("tracking_number", "?")
        tier = data.get("resolution_tier", "?")
        ai_res = data.get("ai_resolution")
        confidence = data.get("confidence_score", 0)
        status = data.get("status", "?")
        dept = data.get("department", "?")
        priority = data.get("priority", "?")

        has_ai = ai_res is not None and len(ai_res) > 10
        verdict = "PASS" if has_ai else "FAIL"

        results.append({
            "title": g["title"],
            "tracking": tracking,
            "tier": tier,
            "has_ai": has_ai,
            "confidence": confidence,
            "status": status,
            "verdict": verdict,
        })

        icon = "PASS" if has_ai else "FAIL"
        print(f"    [{icon}] {tracking} | tier={tier} | confidence={confidence:.2f} | dept={dept} | priority={priority}")
        if has_ai:
            preview = ai_res[:120].replace("\n", " ")
            print(f"    AI: \"{preview}...\"")
        else:
            print(f"    NO AI resolution returned to citizen")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r.get("verdict") == "PASS")
    failed = sum(1 for r in results if r.get("verdict") == "FAIL")
    errors = sum(1 for r in results if r.get("status") == "SUBMIT_FAILED")
    print(f"  Passed: {passed}/{len(TEST_GRIEVANCES)}")
    print(f"  Failed: {failed}/{len(TEST_GRIEVANCES)}")
    if errors:
        print(f"  Errors: {errors}/{len(TEST_GRIEVANCES)}")

    print("\nDetailed results:")
    for r in results:
        if r.get("status") == "SUBMIT_FAILED":
            print(f"  ERROR  {r['title'][:50]}")
        else:
            icon = "PASS" if r["has_ai"] else "FAIL"
            print(f"  {icon:4s}  {r['tracking']}  tier={r['tier']:<16s}  conf={r['confidence']:.2f}  {r['title'][:40]}")

    print("=" * 70)
    if failed > 0:
        print(f"\n!!  {failed} grievance(s) did not return AI resolution to citizen.")
        print("  Possible causes:")
        print("  - Classifier tagged them as officer_action instead of self_resolvable")
        print("  - Confidence score was below threshold (currently 0.60)")
        print("  - Qdrant knowledge base returned no matching context")
    if passed == len(TEST_GRIEVANCES):
        print("\n** All grievances returned AI resolutions to the citizen! **")


if __name__ == "__main__":
    main()
