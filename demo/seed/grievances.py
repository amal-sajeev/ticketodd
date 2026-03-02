# Seed data: Grievances (91 entries exercising every feature)
#
# Dates: created_at spread evenly from 6 months ago to today for charts/reports.
#
# Coverage matrix:
#   Statuses   : resolved (25), in_progress (15), escalated (10), pending (41)
#   Departments: all 8 represented
#   Priorities : low, medium, high, urgent
#   Sentiments : positive, neutral, negative, frustrated
#   Res. tiers : self_resolvable, officer_action, escalation
#   Res. types : ai, manual, hybrid
#   Feedback   : 1, 2, 3, 4, 5 (full range)
#   Special    : is_public, is_anonymous, scheme_match (with eligibility_likely),
#                sub_tasks, impact_score, estimated_resolution_deadline, notes,
#                citizen_phone, multiple languages, location on all entries

from datetime import timedelta

from .config import (
    new_id, now_utc, geojson_point,
    OFFICER_CATS, PRIORITY_SLA_HOURS,
)

# ---------------------------------------------------------------------------
# Grievance records
# ---------------------------------------------------------------------------
GRIEVANCES = [
    # ======================================================================
    # RESOLVED (12)
    # ======================================================================

    # 1 — AI self-resolved, english, low priority, positive feedback
    {"title": "How to check PMAY-G application status online",
     "description": "I want to know how to check my PMAY-G housing application status online. I applied 2 months ago at the GP office. I do not know the website or process. Can someone guide me?",
     "district": "Puri", "department": "rural_housing", "priority": "low",
     "sentiment": "neutral", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "english",
     "ai_resolution": "You can check your PMAY-G application status online through the **Awaas+ portal**:\n\n1. Visit [awaassoft.nic.in](https://awaassoft.nic.in)\n2. Click on **'Stakeholders'** → **'IAY/PMAY-G Beneficiary'**\n3. Enter your **Registration Number** (given at GP office) or search by name\n4. Your application status, installment details, and geo-tag progress will be displayed\n\nAlternatively, you can call the **PMAY-G helpline: 1800-11-6446** (toll-free) for status updates.\n\nIf you don't have your registration number, visit your Gram Panchayat office — the GP Secretary can look it up using your Aadhaar number.",
     "confidence_score": 0.92, "resolution_feedback": 5,
     "citizen_name": "Ganesh Behera", "citizen_email": "ganesh.behera@email.com",
     "citizen_key": "citizen1"},

    # 2 — AI self-resolved, english
    {"title": "What documents are needed for MGNREGS job card application",
     "description": "I want to apply for MGNREGS job card. What documents do I need to bring to the GP office? My neighbor said they need many papers.",
     "district": "Khordha", "department": "mgnregs", "priority": "low",
     "sentiment": "neutral", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "english",
     "ai_resolution": "To apply for a **MGNREGS Job Card**, you need the following documents:\n\n### Required Documents\n1. **Aadhaar Card** of the head of household\n2. **Family photograph** (passport size, with all adult members)\n3. **Address proof** (Aadhaar/Voter ID/Ration card)\n\n### Process\n- Visit your **Gram Panchayat office** and submit a written application\n- The GP Secretary will process it within **15 days**\n- Job card is issued **free of cost** — no fee is required\n- Once issued, you can demand work by giving written application to the Sarpanch\n\n### Important\n- **No income criteria** — every rural household is eligible\n- If the card is not issued within 15 days, complain to the Block MGNREGS Programme Officer\n- Current wage rate in Odisha: **Rs. 289/day** (2025-26)",
     "confidence_score": 0.95, "resolution_feedback": 4,
     "citizen_name": "Prasanna Sahoo", "citizen_email": "prasanna.sahoo@email.com",
     "citizen_key": "citizen1"},

    # 3 — Officer manual, high priority, frustrated, with notes
    {"title": "MGNREGS wages not paid for 60 days in Kalahandi",
     "description": "I and 30 other workers from our village completed road construction work under MGNREGS 60 days ago. Muster rolls were signed. But wages have not been credited to our bank accounts. The GP Sarpanch says he does not know the reason.",
     "district": "Kalahandi", "department": "mgnregs", "priority": "high",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english",
     "manual_resolution": "**Investigation completed by Block MGNREGS Programme Officer.**\n\nThe Fund Transfer Order (FTO) was stuck due to incorrect bank account details for 23 out of 31 workers. The account numbers had data entry errors at the GP level.\n\n**Actions taken:**\n1. Aadhaar-based re-verification of all 31 worker bank accounts completed\n2. Corrected FTO resubmitted to State MGNREGS cell\n3. Wages for 60 days (**Rs. 17,340 per worker**) transferred on 28-Jan-2026\n4. Delay compensation @ 0.05% per day also credited\n5. GP Secretary issued warning for negligence in data entry\n\nAll 31 workers have confirmed receipt of wages. Case closed.",
     "assigned_officer": "Sri Bikram Sahu, MGNREGS PO", "resolution_feedback": 5,
     "notes": [
         {"officer": "Sri Bikram Sahu, MGNREGS PO", "content": "FTO investigation started. Found bank account mismatch for 23 workers.", "note_type": "internal"},
         {"officer": "Sri Bikram Sahu, MGNREGS PO", "content": "All accounts corrected via Aadhaar re-verification. FTO resubmitted.", "note_type": "citizen_facing"},
         {"officer": "Sri Bikram Sahu, MGNREGS PO", "content": "Wages credited to all 31 accounts. Delay compensation included.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Laxman Naik", "citizen_email": "laxman.naik@email.com",
     "citizen_key": "citizen3"},

    # 4 — Officer manual, urgent, tribal area, is_public
    {"title": "Bore well non-functional for 4 months in tribal village",
     "description": "The only bore well in our tribal hamlet of 60 families in Mayurbhanj has not been working for 4 months. The hand pump handle is broken and motor is burned out. Women and children walk 3 km daily to fetch water from a stream.",
     "district": "Mayurbhanj", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english", "is_public": True,
     "manual_resolution": "**RWSS Junior Engineer inspection report:**\n\nSite inspection conducted on 05-Jan-2026. Found:\n- Submersible pump motor completely burned out (beyond repair)\n- Hand pump handle fractured at pivot joint\n\n**Repair actions:**\n1. New 1.5 HP submersible pump motor installed under Basudha maintenance budget — **completed 08-Jan-2026**\n2. Hand pump handle replaced with heavy-duty stainless steel model — **completed 08-Jan-2026**\n3. GP Jalasathi trained on basic pump maintenance and monthly inspection checklist\n4. Quarterly preventive maintenance schedule established with RWSS Block office\n\nWater supply restored to all 60 households. Follow-up visit scheduled for March 2026.",
     "assigned_officer": "Er. Anil Panigrahi, EE-RWSS", "resolution_feedback": 5,
     "impact_score": 89,
     "notes": [
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Urgent: Assigned RWSS JE for immediate site inspection given 60 families affected.", "note_type": "internal"},
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Repair completed. Water supply restored. Jalasathi training done.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Gurubari Hansda", "citizen_email": "gurubari.hansda@email.com",
     "citizen_key": "citizen3"},

    # 5 — Hybrid resolution, sanitation, is_public
    {"title": "SBM toilet subsidy not received after construction completed",
     "description": "I constructed a toilet under SBM-G scheme 5 months ago. The GP Secretary took photos and said the incentive of Rs. 12,000 will be credited. But nothing has been received. When I ask, they say it is under processing.",
     "district": "Balasore", "department": "sanitation", "priority": "medium",
     "sentiment": "negative", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "hybrid", "language": "english", "is_public": True,
     "ai_resolution": "Based on similar resolved cases, the most common cause of delayed SBM-G incentive payments is that the beneficiary was not registered in the IMIS (Information Management and Monitoring System) portal. The GP Secretary needs to complete the online entry with photo evidence before payment can be processed.",
     "manual_resolution": "**Block SBM Coordinator verified the complaint.**\n\nInvestigation confirmed the AI assessment — beneficiary construction was verified physically but the IMIS portal entry was incomplete. GP Secretary had uploaded photos but did not complete the verification form.\n\n**Resolution:**\n1. IMIS entry completed by Block SBM Coordinator on 15-Jan-2026\n2. Incentive of **Rs. 12,000** credited to beneficiary bank account (A/C ending 4521) on 22-Jan-2026\n3. GP Secretary counseled on proper IMIS documentation procedure\n\nPayment confirmed by beneficiary.",
     "assigned_officer": "Smt. Sarojini Das, Block Sanitation Coord.", "confidence_score": 0.78,
     "resolution_feedback": 4, "impact_score": 42,
     "scheme_match": {"scheme_name": "Swachh Bharat Mission - Gramin (SBM-G)", "relevance_score": 0.91,
                      "eligibility_likely": True, "eligibility_reasoning": "Grievance directly relates to SBM-G IHHL incentive payment delay."},
     "notes": [
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "AI draft was accurate — IMIS entry was indeed incomplete. Approved and supplemented with specific resolution details.", "note_type": "internal"},
     ],
     "citizen_name": "Basanti Jena", "citizen_email": "basanti.jena@email.com",
     "citizen_key": "citizen2"},

    # 6 — Officer manual, urgent, water quality, with notes
    {"title": "Water from JJM tap is yellow and smells bad",
     "description": "We received JJM tap connections 2 months ago in our village in Dhenkanal. But the water is yellowish with a metallic smell. Children have been getting stomach problems. We suspect high iron content. Please test the water quality.",
     "district": "Dhenkanal", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english",
     "manual_resolution": "**Water quality test results and remediation:**\n\nRWSS collected samples from 5 points in the pipeline network on 10-Jan-2026.\n\n**Lab Results:**\n| Parameter | Found | Safe Limit |\n|-----------|-------|------------|\n| Iron | 3.2 mg/L | 0.3 mg/L |\n| Turbidity | 8 NTU | 5 NTU |\n| pH | 6.8 | 6.5-8.5 |\n| Bacteriological | Safe | — |\n\n**High iron content confirmed (10x above safe limit).**\n\n**Remediation:**\n1. Iron Removal Plant (IRP) ordered under JJM water quality component — **installed 28-Jan-2026**\n2. Pipeline flushing done to clear accumulated deposits\n3. Follow-up water testing on 02-Feb-2026 confirmed iron levels at **0.2 mg/L** (within safe limits)\n4. Quarterly water quality monitoring schedule established\n\nWater is now safe for consumption. Residents advised to report any recurrence immediately.",
     "assigned_officer": "Er. Anil Panigrahi, EE-RWSS", "resolution_feedback": 5,
     "impact_score": 76,
     "notes": [
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "URGENT: Water samples collected from 5 points. Sent to district RWSS lab for testing.", "note_type": "internal"},
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Lab results confirm high iron. IRP ordered under JJM water quality budget.", "note_type": "citizen_facing"},
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "IRP installed and operational. Follow-up test confirms safe water quality.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Surekha Pradhan", "citizen_email": "surekha.pradhan@email.com",
     "citizen_key": "citizen2"},

    # 7 — AI self-resolved, hindi language
    {"title": "MGNREGS mein kaam ka demand kaise kare",
     "description": "Mera job card ban gaya hai lekin mujhe kaam nahi mil raha. GP office mein koi batata nahi ki kaise demand karna hai. Kya process hai?",
     "district": "Nuapada", "department": "mgnregs", "priority": "low",
     "sentiment": "neutral", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "hindi",
     "ai_resolution": "## MGNREGS mein kaam ka demand kaise karein\n\n### Tarika\n1. **Likha hua aavedan** Gram Panchayat Sarpanch ko dijiye\n2. Aavedan mein likhein: \"Main MGNREGS ke tahat kaam maangta/maangti hoon\"\n3. **Job Card number** aur **tarikh** zaroor likhein\n4. GP ko **15 din** ke andar kaam dena hoga\n5. Agar 15 din mein kaam nahi milta, toh **berojgaari bhatta** milega\n\n### Zaruri baatein\n- Kaam ka demand **likhi mein** hona chahiye\n- GP Secretary se **raseed** zaroor lein\n- Mazdoori: **Rs. 289/din** (Odisha, 2025-26)\n- Shikayat: Block MGNREGS Programme Officer se karein",
     "confidence_score": 0.90, "resolution_feedback": 4,
     "citizen_name": "Ratan Majhi", "citizen_email": "ratan.majhi@email.com",
     "citizen_key": "citizen3"},

    # 8 — Officer manual, panchayati_raj
    {"title": "Gram Panchayat not holding Gram Sabha for over 1 year",
     "description": "Our Gram Panchayat in Boudh district has not conducted any Gram Sabha since January 2025. Neither the mandatory Republic Day nor Gandhi Jayanti sessions were held. The Sarpanch is not responsive to requests. Beneficiary lists are being decided without public consultation.",
     "district": "Boudh", "department": "panchayati_raj", "priority": "high",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english", "is_public": True,
     "manual_resolution": "**BDO conducted inquiry under Odisha Gram Panchayat Act.**\n\nShow-cause notice served to Sarpanch within 7 days. Gram Sabha scheduled within 15 days with BDO personally supervising. Ward members notified through public notice and mobile alerts. Gram Sabha held successfully on 20-Jan-2026 with quorum met.\n\nRegular biannual schedule enforced going forward. Sarpanch warned: failure to hold next scheduled Gram Sabha will result in formal proceedings under Section 12A of the Act.",
     "assigned_officer": "Smt. Priya Pattnaik, BDO", "resolution_feedback": 4,
     "impact_score": 55,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Show-cause notice issued to Sarpanch. Gram Sabha scheduled within 15 days.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Brundaban Sahu", "citizen_email": "brundaban.sahu@email.com",
     "citizen_key": "citizen1"},

    # 9 — AI self-resolved, odia language, low priority
    {"title": "SBM-G re toilet baniba pain ki kagajpatra darkar",
     "description": "Mu SBM-G yojana re toilet banibaku chahuchhi. GP office re jibaku kahile kana kana lagiba se kahile nahin. Daya kari janaantu ki ki document lagiba.",
     "district": "Nayagarh", "department": "sanitation", "priority": "low",
     "sentiment": "neutral", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "odia",
     "ai_resolution": "## SBM-G Toilet Nirman Pain Aabashyak Kagajpatra\n\n### Darkar Heba:\n1. **Aadhaar Card** — paribar mukhya nka\n2. **Bank passbook** — incentive jama pain\n3. **BPL certificate** (jadi applicable)\n4. **GP office re aavedan** — likhi kar diantu\n\n### Prakriya:\n- GP Secretary verify karibe eligibility\n- Toilet baniba pare **photo evidence** diantu\n- **Rs. 12,000** incentive bank account re jama heba\n- Kono fee lagibani — free of cost\n\n### Sahayata:\n- Block SBM Coordinator nku call karantu\n- GP Sarpanch nku bhetantu",
     "confidence_score": 0.87, "resolution_feedback": 5,
     "scheme_match": {"scheme_name": "Swachh Bharat Mission - Gramin (SBM-G)", "relevance_score": 0.94,
                      "eligibility_likely": True, "eligibility_reasoning": "Citizen is asking about SBM-G toilet construction documents — direct scheme match."},
     "citizen_name": "Sushant Sahoo", "citizen_email": "sushant.sahoo@email.com",
     "citizen_key": "citizen1"},

    # 10 — Hybrid, rural_livelihoods, positive sentiment
    {"title": "Mission Shakti SHG revolving fund disbursed after long delay",
     "description": "Our SHG 'Maa Lakshmi' in Gajapati finally received the revolving fund after filing this grievance. Thank you for the help. The Block OLM Coordinator was very helpful.",
     "district": "Gajapati", "department": "rural_livelihoods", "priority": "medium",
     "sentiment": "positive", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "hybrid", "language": "english",
     "ai_resolution": "Based on service memory, SHGs that have not received their revolving fund often need a grading exercise. Recommend Block OLM Coordinator to conduct grading and process the RF disbursement.",
     "manual_resolution": "Block OLM Coordinator conducted grading exercise. SHG 'Maa Lakshmi' qualified for Grade-I rating. Revolving Fund of Rs. 15,000 disbursed within 10 days through NRLM fund flow. SHG also linked to bank for future credit linkage.",
     "assigned_officer": "Sri Ranjit Mishra, DRDA PD", "confidence_score": 0.80,
     "resolution_feedback": 5, "impact_score": 38,
     "citizen_name": "Kuni Sabar", "citizen_email": "kuni.sabar@email.com",
     "citizen_key": "citizen4"},

    # 11 — Officer manual, infrastructure, with sub_tasks (cross-department)
    {"title": "Waterlogged road and clogged drain causing health hazard in Jajpur GP",
     "description": "The main village road in Jajpur GP is permanently waterlogged because the roadside drain is clogged with solid waste. Mosquito breeding has increased. Children are falling sick with dengue. We need both drain cleaning and road repair.",
     "district": "Jajpur", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english", "is_public": True,
     "manual_resolution": "**Multi-department coordinated response:**\n\n1. **Infrastructure**: Road surface repaired and camber corrected for drainage. Cost Rs. 2.1 lakh under FC untied grants.\n2. **Sanitation**: Drain cleaned and SLWM unit activated. Weekly solid waste collection started.\n\nHealth department also conducted fogging for mosquito control. Follow-up inspection scheduled in 30 days.",
     "assigned_officer": "Sri Debashis Swain, Sr. District Officer", "resolution_feedback": 4,
     "impact_score": 67,
     "sub_tasks": [
         {"id": "st-jajpur-1", "department": "infrastructure", "task": "Repair waterlogged road surface and correct camber for proper drainage",
          "status": "resolved", "assigned_officer": "Sri Debashis Swain, Sr. District Officer"},
         {"id": "st-jajpur-2", "department": "sanitation", "task": "Clean clogged drain, activate SLWM unit, start weekly waste collection",
          "status": "resolved", "assigned_officer": "Smt. Sarojini Das, Block Sanitation Coord."},
     ],
     "notes": [
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "Cross-department case: infrastructure + sanitation. Coordinating with Block Sanitation.", "note_type": "internal"},
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "Road repair and drain cleaning completed. Health dept fogging done.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Lalita Sahoo", "citizen_email": "lalita.sahoo@email.com",
     "citizen_key": "citizen2"},

    # 12 — Officer manual, rural_housing, scheme_match
    {"title": "Nirman Shramik housing application rejected without reason",
     "description": "I am a registered construction worker in Sambalpur. I applied for Nirman Shramik Pucca Ghar Yojana 4 months ago. The GP Secretary says my application was rejected but cannot tell me why.",
     "district": "Sambalpur", "department": "rural_housing", "priority": "medium",
     "sentiment": "negative", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english",
     "manual_resolution": "Block office verified applicant's registration with Building Workers Welfare Board. Found registration had expired 2 months before application. Helped applicant renew registration at Labour office (Rs. 25 fee). Fresh housing application submitted through GP and accepted. Construction assistance sanctioned within 30 days.",
     "assigned_officer": "Smt. Lopamudra Jena, BDO Housing", "resolution_feedback": 4,
     "scheme_match": {"scheme_name": "Nirman Shramik Pucca Ghar Yojana", "relevance_score": 0.96,
                      "eligibility_likely": True, "eligibility_reasoning": "Direct application under Nirman Shramik housing scheme for construction workers."},
     "citizen_name": "Bhagirathi Suna", "citizen_email": "bhagirathi.suna@email.com",
     "citizen_key": "citizen3"},

    # ======================================================================
    # IN PROGRESS (8)
    # ======================================================================

    # 13 — In progress, urgent, water supply, with notes + estimated deadline
    {"title": "No JJM tap water connection despite being in village action plan",
     "description": "Our habitation of 85 families in Koraput block was included in the JJM Village Action Plan last year. Pipeline work started but stopped 2 months ago. No tap connections provided yet. We still depend on a contaminated open well.",
     "district": "Koraput", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english", "is_public": True,
     "assigned_officer": "Er. Anil Panigrahi, EE-RWSS",
     "ai_resolution": "Based on similar cases in Koraput district, pipeline work stoppages are commonly caused by: (1) rocky terrain requiring percussion drilling, (2) contractor payment disputes, or (3) Right-of-Way clearance issues. Recommend contacting the Block RWSS Executive Engineer for specific status on the pipeline extension work.",
     "confidence_score": 0.65, "impact_score": 82,
     "estimated_resolution_days": 30,
     "notes": [
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Contacted RWSS EE Koraput. Contractor reports rock formations requiring special equipment. Percussion drilling team being mobilized.", "note_type": "internal"},
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Your complaint has been escalated to the RWSS Executive Engineer, Koraput. Special drilling equipment is being arranged to complete the pipeline through rocky terrain. Expected completion: 4-6 weeks.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Dambaru Majhi", "citizen_email": "dambaru.majhi@email.com",
     "citizen_key": "citizen3"},

    # 14 — In progress, high, infrastructure, with notes
    {"title": "BGBO road project abandoned halfway in Rayagada block",
     "description": "A 3 km rural road sanctioned under BGBO from our village to the main road has been abandoned after only 1 km of construction. The contractor left the site 4 months ago. During monsoon the unfinished road becomes impassable and dangerous.",
     "district": "Rayagada", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Sri Debashis Swain, Sr. District Officer",
     "impact_score": 61,
     "estimated_resolution_days": 45,
     "notes": [
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "DPO field inspection completed. Contractor abandoned at 60% citing material cost escalation. Contract termination and penalty proceedings initiated.", "note_type": "internal"},
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "The contractor's performance has been found unsatisfactory. Contract termination process has been started and re-tendering will begin shortly. A new contractor will be engaged to complete the remaining 2 km.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Kamala Sabar", "citizen_email": "kamala.sabar@email.com",
     "citizen_key": "citizen4"},

    # 15 — In progress, urgent, infrastructure, SLA near-breach
    {"title": "Damaged culvert blocking school access in Kandhamal village",
     "description": "A culvert on the village road in Kandhamal collapsed during heavy rains last month. Now children cannot cross to reach the school on the other side. Vehicles also cannot pass. We need emergency repair under BGBO or Finance Commission grants.",
     "district": "Kandhamal", "department": "infrastructure", "priority": "urgent",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english", "is_public": True,
     "assigned_officer": "Sri Debashis Swain, Sr. District Officer",
     "impact_score": 74,
     "estimated_resolution_days": 60,
     "notes": [
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "Emergency: Temporary bailey bridge placed within 48 hours for vehicle and pedestrian access. Permanent culvert reconstruction sanctioned under BGBO with Rs. 8 lakh estimate.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Gobinda Kanhar", "citizen_email": "gobinda.kanhar@email.com",
     "citizen_key": "citizen3"},

    # 16 — In progress, medium, rural_housing, scheme_match
    {"title": "PMAY-G second installment not released despite completing lintel",
     "description": "I completed construction up to lintel level for my PMAY-G house 3 months ago. Geo-tagged photos were taken by the Block TA. But second installment of Rs. 40,000 has not been released. I have exhausted my savings and cannot continue construction.",
     "district": "Ganjam", "department": "rural_housing", "priority": "high",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Smt. Lopamudra Jena, BDO Housing",
     "impact_score": 52,
     "estimated_resolution_days": 15,
     "scheme_match": {"scheme_name": "PMAY-Gramin (Pradhan Mantri Awas Yojana - Gramin)", "relevance_score": 0.97,
                      "eligibility_likely": True, "eligibility_reasoning": "Grievance is about delayed PMAY-G installment release at lintel stage."},
     "notes": [
         {"officer": "Smt. Lopamudra Jena, BDO Housing", "content": "Block TA visited. Geo-tag photo was rejected by Awaas+ portal due to GPS inaccuracy. Re-photographing scheduled.", "note_type": "internal"},
     ],
     "citizen_name": "Somanath Behera", "citizen_email": "somanath.behera@email.com",
     "citizen_key": "citizen1"},

    # 17 — In progress, medium, mgnregs, sub_tasks (cross-dept)
    {"title": "MGNREGS work demand for pond renovation but land encroached",
     "description": "We demanded pond renovation work under MGNREGS in Balangir. But the GP says the pond land is encroached by a private person. We need both the encroachment removed and the pond renovated for irrigation.",
     "district": "Balangir", "department": "mgnregs", "priority": "medium",
     "sentiment": "negative", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Sri Bikram Sahu, MGNREGS PO",
     "impact_score": 45,
     "estimated_resolution_days": 40,
     "sub_tasks": [
         {"id": "st-balangir-1", "department": "panchayati_raj", "task": "Serve notice to encroacher and initiate eviction proceedings under GP Act",
          "status": "in_progress", "assigned_officer": "Smt. Priya Pattnaik, BDO"},
         {"id": "st-balangir-2", "department": "mgnregs", "task": "Prepare pond renovation estimate and muster roll after land is cleared",
          "status": "pending", "assigned_officer": "Sri Bikram Sahu, MGNREGS PO"},
     ],
     "citizen_name": "Harihar Sahu", "citizen_email": "harihar.sahu@email.com",
     "citizen_key": "citizen1"},

    # 18 — In progress, high, sanitation, SLA breached (old created_at)
    {"title": "ODF Plus village losing status — no waste collection for 3 months",
     "description": "Our village in Kendrapara achieved ODF Plus status last year. But the solid waste collection system has completely stopped. Waste is piling up. Plastic is being burned openly. We will lose our ODF Plus status if this continues.",
     "district": "Kendrapara", "department": "sanitation", "priority": "high",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Smt. Sarojini Das, Block Sanitation Coord.",
     "impact_score": 58,
     "sla_breached": True,
     "notes": [
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "Investigating SLWM unit status. Operator had left due to non-payment. Arranging replacement.", "note_type": "internal"},
     ],
     "citizen_name": "Pradeep Rout", "citizen_email": "pradeep.rout@email.com",
     "citizen_key": "citizen2"},

    # 19 — In progress, medium, rural_livelihoods
    {"title": "NRLM SHG bank linkage loan pending for 8 months",
     "description": "Our SHG 'Maa Tarini' in Nabarangpur has been regularly saving and meeting for 2 years. We applied for bank linkage loan of Rs. 2 lakh 8 months ago. The bank keeps asking for more documents. The Block OLM office says they have forwarded it.",
     "district": "Nabarangpur", "department": "rural_livelihoods", "priority": "medium",
     "sentiment": "negative", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Sri Ranjit Mishra, DRDA PD",
     "ai_resolution": "Based on similar resolved cases in Nabarangpur district, the most common reason for delayed SHG bank linkage is incomplete grading documentation. The Block OLM Coordinator should:\n\n1. Verify that the SHG credit rating/grading exercise has been completed\n2. Check if monthly meeting minutes are properly documented for all 24 months\n3. Coordinate directly with the bank branch manager to identify specific pending documents\n4. If the bank is non-cooperative, escalate to the Lead District Manager (LDM) under NRLM banking correspondent guidelines\n\nTypical resolution timeline after intervention: 15-20 days.",
     "confidence_score": 0.72,
     "estimated_resolution_days": 20,
     "citizen_name": "Sabita Nayak", "citizen_email": "sabita.nayak@email.com",
     "citizen_key": "citizen4"},

    # 20 — In progress, high, panchayati_raj, odia language
    {"title": "Sarpanch nku no-confidence motion pain sahayata chahantu",
     "description": "Amara GP ra Sarpanch 2 barsha re kichhi kaam karani nahin. Fund misuse karuchhanti. Amme 2/3 ward member nku ekatrika karichhun. No-confidence motion file karibaku chahuchhu kintu prakriya januni.",
     "district": "Deogarh", "department": "panchayati_raj", "priority": "high",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "odia",
     "assigned_officer": "Smt. Priya Pattnaik, BDO",
     "impact_score": 48,
     "estimated_resolution_days": 30,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Verified: 8 out of 11 ward members support no-confidence motion. Guiding them through formal process under Odisha GP Act Sec 10.", "note_type": "internal"},
     ],
     "citizen_name": "Sukanta Mahapatra", "citizen_email": None,
     "citizen_key": "citizen1"},

    # ======================================================================
    # ESCALATED (5)
    # ======================================================================

    # 21 — Escalated, anonymous, corruption
    {"title": "Finance Commission grant funds allegedly misused by Sarpanch",
     "description": "In our GP in Kendrapara, the Sarpanch has used Finance Commission untied grant money to construct a boundary wall around his own property. The community demanded a road and drain but was ignored. Villagers have proof including photos and payment vouchers.",
     "district": "Kendrapara", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english",
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Complaint forwarded to DPO for audit. Photographs and voucher copies received from complainant.", "note_type": "internal"},
     ],
     "citizen_name": None, "citizen_email": None, "is_anonymous": True,
     "citizen_key": "citizen1"},

    # 22 — Escalated, anonymous, corruption, urgent
    {"title": "Sarpanch demanding bribe for PMAY-G beneficiary selection",
     "description": "The Sarpanch of our GP in Malkangiri is demanding Rs. 10,000 from each family to include them in the PMAY-G beneficiary list. Many deserving families who cannot pay are being left out. This is corruption and must be investigated.",
     "district": "Malkangiri", "department": "rural_housing", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english", "is_public": True,
     "impact_score": 91,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Serious corruption allegation. Referred to District Collector and Vigilance cell for investigation. PMAY-G beneficiary list frozen pending inquiry.", "note_type": "internal"},
     ],
     "citizen_name": None, "citizen_email": None, "is_anonymous": True,
     "citizen_key": "citizen3"},

    # 23 — Escalated, mgnregs, death at worksite
    {"title": "MGNREGS worker death at worksite — no compensation paid",
     "description": "A MGNREGS worker died due to wall collapse at a pond deepening worksite in Koraput 2 months ago. The family has not received any compensation. The GP says it is not their responsibility. The Block office is not responding.",
     "district": "Koraput", "department": "mgnregs", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english", "is_public": True,
     "impact_score": 95,
     "notes": [
         {"officer": "Sri Bikram Sahu, MGNREGS PO", "content": "Workplace death confirmed. MGNREGS guidelines mandate Rs. 5 lakh ex-gratia. File forwarded to District Programme Coordinator for immediate sanction.", "note_type": "internal"},
         {"officer": "Sri Bikram Sahu, MGNREGS PO", "content": "Investigation underway. MGNREGS guidelines provide for Rs. 5 lakh compensation for worksite death. Matter has been escalated to the District Programme Coordinator.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Sukra Majhi", "citizen_email": "sukra.majhi@email.com",
     "citizen_key": "citizen3"},

    # 24 — Escalated, rural_water_supply, systemic
    {"title": "Entire block JJM pipeline non-functional — contractor absconded",
     "description": "The JJM pipeline contractor for our entire block in Rayagada has absconded after receiving 70% payment. Pipeline laid in 12 villages is non-functional — joints are leaking, fittings are substandard. Over 5,000 families affected.",
     "district": "Rayagada", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english",
     "impact_score": 97,
     "notes": [
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Block-level JJM failure. Contractor absconded. FIR filed. Bank guarantee being invoked. Emergency water supply via tankers arranged for worst-affected villages.", "note_type": "internal"},
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Emergency measures activated: (1) Water tanker supply for 12 villages, (2) FIR filed against contractor, (3) New contractor being engaged for repair works.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Padman Sabar", "citizen_email": "padman.sabar@email.com",
     "citizen_key": "citizen4",
     "systemic_link": True},

    # 25 — Escalated, panchayati_raj, large-scale
    {"title": "Panchayat Samiti members colluding to block legitimate GP projects",
     "description": "Multiple GP Sarpanches from Ganjam block report that the Panchayat Samiti chairman is deliberately blocking BGBO and FC project approvals for GPs that did not support his election. 15 GPs are affected with Rs. 8 crore of approved projects stuck.",
     "district": "Ganjam", "department": "panchayati_raj", "priority": "high",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english",
     "impact_score": 85,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Multiple Sarpanch complaints verified. Matter referred to District Collector and State PR Department for inquiry into Samiti chairman conduct.", "note_type": "internal"},
     ],
     "citizen_name": "Krushna Chandra Swain", "citizen_email": "krushna.swain@email.com",
     "citizen_key": "citizen1"},

    # ======================================================================
    # PENDING (18)
    # ======================================================================

    # 26 — Pending, with AI draft, infrastructure
    {"title": "BGBO community hall construction using substandard material",
     "description": "A community hall is being built under BGBO scheme in our village in Sundargarh. The contractor is using poor quality bricks and less cement. Cracks have already appeared in the walls. We reported to the Block JE but no action taken.",
     "district": "Sundargarh", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "escalation",
     "language": "english",
     "ai_resolution": "This is a serious quality control issue that requires immediate field investigation. Recommended actions:\n\n1. **Block JE must conduct a quality assessment** including concrete cube test and material verification\n2. If substandard material is confirmed, the contractor should be directed to **demolish and rebuild** the affected portion at their own cost\n3. A **penalty of 10% should be deducted** from the contractor's payment as per BGBO guidelines\n4. **Third-party quality monitoring** should be engaged for the remaining work\n\nThis matter should be escalated to the District Panchayat Officer if the Block JE does not act within 7 days.",
     "confidence_score": 0.68,
     "citizen_name": "Mangal Kisan", "citizen_email": "mangal.kisan@email.com",
     "citizen_key": "citizen2"},

    # 27 — Pending, self-resolvable, panchayati_raj
    {"title": "How to register complaint about non-functional street light",
     "description": "The solar street lights installed by GP in our village stopped working 2 weeks ago. Where do I complain? Whom should I contact?",
     "district": "Cuttack", "department": "panchayati_raj", "priority": "low",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "english",
     "ai_resolution": "For non-functional solar street lights installed by the Gram Panchayat, here's how to get them fixed:\n\n### Step 1: Contact GP Office\n- Report to your **GP Secretary** or **Sarpanch** in writing\n- Solar street lights installed by GP are maintained under **Finance Commission grants**\n\n### Step 2: Warranty Check\n- Most solar street lights have a **5-year warranty** from the manufacturer\n- Ask the GP Secretary to check if the lights are still under warranty\n- If under warranty, the manufacturer must repair/replace at no cost\n\n### Step 3: Escalation\n- If GP doesn't act within 7 days, contact the **Block Development Officer (BDO)**\n- BDO contact details available at [panchayat.odisha.gov.in](https://panchayat.odisha.gov.in)\n\n### Step 4: Alternative Reporting\n- Call the **PR&DW Department helpline** for your district\n- You can also report on the **e-Gram Swaraj portal** (egramswaraj.gov.in)",
     "confidence_score": 0.88,
     "citizen_name": "Sudhir Mohanty", "citizen_email": "sudhir.mohanty@email.com",
     "citizen_key": "citizen1"},

    # 28 — Pending, water supply, high priority
    {"title": "JJM pipeline laid but no water flowing for 2 months",
     "description": "JJM pipeline was laid to our habitation in Angul 2 months ago and taps were installed in all houses. But no water has ever flowed through the taps. The overhead tank was built but seems not connected. We still use the old tube well.",
     "district": "Angul", "department": "rural_water_supply", "priority": "high",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Bijay Kumar Sahu", "citizen_email": "bijay.sahu@email.com",
     "citizen_key": "citizen2"},

    # 29 — Pending, mgnregs, medium
    {"title": "MGNREGS job card not issued despite applying 3 months ago",
     "description": "I applied for a MGNREGS Job Card at the GP office in Nuapada 3 months ago with all required documents including family photo and Aadhaar. The GP Secretary says it is under processing. Without the card, I cannot demand work or earn wages.",
     "district": "Nuapada", "department": "mgnregs", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Ratan Majhi", "citizen_email": "ratan.majhi@email.com",
     "citizen_key": "citizen3"},

    # 30 — Pending, water supply, medium
    {"title": "Request for new bore well under Basudha in Sundargarh village",
     "description": "Our village of 150 families in Sundargarh has only one hand pump which dries up in summer. We face severe water crisis from March to June every year. We request installation of a new bore well under Basudha scheme. The GP has passed a resolution.",
     "district": "Sundargarh", "department": "rural_water_supply", "priority": "medium",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "scheme_match": {"scheme_name": "Basudha (Buxi Jagabandhu Assured Water Supply to Habitations)",
                      "relevance_score": 0.88,
                      "eligibility_likely": True, "eligibility_reasoning": "Citizen is requesting bore well under Basudha scheme. Direct scheme match for water supply to unserved habitation."},
     "citizen_name": "Birsa Munda", "citizen_email": "birsa.munda@email.com",
     "citizen_key": "citizen3"},

    # 31 — Pending, sanitation, medium
    {"title": "SBM community waste management not functioning in Jajpur GP",
     "description": "A Solid and Liquid Waste Management (SLWM) unit was installed in our GP under SBM-G last year. It has not been operational since installation. Waste is piling up near the unit. No staff assigned for operation. The village is losing its ODF Plus status.",
     "district": "Jajpur", "department": "sanitation", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Lalita Sahoo", "citizen_email": "lalita.sahoo@email.com",
     "citizen_key": "citizen2"},

    # 32 — Pending, mgnregs, worksite facilities
    {"title": "MGNREGS worksite has no shade or drinking water facility",
     "description": "At the MGNREGS worksite near our village in Bargarh, there is no shade shelter, no drinking water, and no first-aid kit. Women workers are suffering in the heat. The mate says there is no budget for facilities. This violates MGNREGS guidelines.",
     "district": "Bargarh", "department": "mgnregs", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Pramila Sahu", "citizen_email": "pramila.sahu@email.com",
     "citizen_key": "citizen4"},

    # 33 — Pending, rural_livelihoods, medium
    {"title": "Mission Shakti SHG not receiving revolving fund in Gajapati",
     "description": "Our SHG 'Maa Lakshmi' in Gajapati was formed under Mission Shakti 1 year ago. We have been saving Rs. 100/month regularly and conducting weekly meetings. But the Revolving Fund of Rs. 15,000 has not been received. The Block OLM Coordinator says funds are exhausted.",
     "district": "Gajapati", "department": "rural_livelihoods", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Kuni Sabar", "citizen_email": "kuni.sabar@email.com",
     "citizen_key": "citizen4"},

    # 34 — Pending, rural_housing, SLA breached
    {"title": "PMAY-G house foundation cracked within 6 months of construction",
     "description": "My PMAY-G house in Kalahandi was completed 6 months ago. The foundation has developed major cracks and the walls are showing signs of settling. The contractor used poor quality material. I need the house repaired or rebuilt.",
     "district": "Kalahandi", "department": "rural_housing", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "sla_breached": True,
     "citizen_name": "Mithun Naik", "citizen_email": "mithun.naik@email.com",
     "citizen_key": "citizen3"},

    # 35 — Pending, water supply, hindi language
    {"title": "Hamare gaon mein paani mein fluoride hai — bacche beemar ho rahe hain",
     "description": "Hamare gaon Jharsuguda mein bore well ka paani mein fluoride bahut zyada hai. Bachchon ke daant kharab ho rahe hain aur haddi mein dard ho raha hai. Doctor ne kaha ki ye fluorosis hai. Hum surakshit paani maangte hain.",
     "district": "Jharsuguda", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "language": "hindi",
     "citizen_name": "Ramesh Patel", "citizen_email": "ramesh.patel@email.com",
     "citizen_key": "citizen1"},

    # 36 — Pending, infrastructure, medium
    {"title": "GP school building roof leaking during monsoon — children affected",
     "description": "The GP primary school in Bhadrak has a leaking roof. During rains, water enters the classroom. Children are getting wet and falling sick. The Sarpanch says there is no fund for repair. Can BGBO or FC grants be used?",
     "district": "Bhadrak", "department": "infrastructure", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Sabitri Das", "citizen_email": "sabitri.das@email.com",
     "citizen_key": "citizen2"},

    # 37 — Pending, panchayati_raj, low
    {"title": "Ward member election result challenged — demand for recount",
     "description": "In the recent Panchayat election in Jagatsinghpur, the ward member election result was declared with a margin of only 3 votes. I and other voters believe there were counting errors. We want a recount or re-election.",
     "district": "Jagatsinghpur", "department": "panchayati_raj", "priority": "low",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Bijay Ranjan Rath", "citizen_email": "bijay.rath@email.com",
     "citizen_key": "citizen1"},

    # 38 — Pending, sanitation, odia language
    {"title": "Amara GP re ODF status jhuth — bahut lok khula re jauchanti",
     "description": "Amara GP Subarnapur re ODF ghoshana karichi kintu asthiti tike bhinna. Bahut paribar re toilet achhi kintu byabahar karuchanti nahin. Khula re shaucha chalichi. SBM team aasi dekhantu.",
     "district": "Subarnapur", "department": "sanitation", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "odia",
     "citizen_name": "Niranjan Meher", "citizen_email": None,
     "citizen_key": "citizen1"},

    # 39 — Pending, rural_livelihoods, medium, sub_tasks
    {"title": "SHG members trained but no market linkage for products in Kendujhar",
     "description": "Our SHG in Kendujhar received skill training for leaf plate making under OLM. We produce 500 plates daily but have no buyer. The Block office promised market linkage 6 months ago but nothing happened. We also need MGNREGS convergence for a drying shed.",
     "district": "Kendujhar", "department": "rural_livelihoods", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "sub_tasks": [
         {"id": "st-kendujhar-1", "department": "rural_livelihoods", "task": "Arrange market linkage through ORMAS/tribal cooperative for leaf plate products",
          "status": "pending", "assigned_officer": None},
         {"id": "st-kendujhar-2", "department": "mgnregs", "task": "Sanction drying shed construction under MGNREGS-OLM convergence",
          "status": "pending", "assigned_officer": None},
     ],
     "citizen_name": "Mamata Nayak", "citizen_email": "mamata.nayak@email.com",
     "citizen_key": "citizen4"},

    # 40 — Pending, rural_water_supply, SLA breached, anonymous
    {"title": "Contractor dumping construction waste into village water source",
     "description": "A BGBO road contractor in Koraput is dumping construction debris into the stream that is our main water source. The water has become muddy and undrinkable. GP and Block officials are aware but taking no action. The contractor is politically connected.",
     "district": "Koraput", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "escalation",
     "language": "english", "is_anonymous": True,
     "sla_breached": True,
     "citizen_name": None, "citizen_email": None,
     "citizen_key": "citizen3"},

    # 41 — Pending, general, low
    {"title": "Request for information on all schemes available for tribal households",
     "description": "I am from a tribal family in Mayurbhanj. I want to know all the government schemes we are eligible for — housing, water, employment, livelihood, education. Please provide a comprehensive list.",
     "district": "Mayurbhanj", "department": "general", "priority": "low",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "self_resolvable",
     "language": "english",
     "ai_resolution": "Here are the key schemes available for tribal households in Odisha:\n\n### Housing\n- **PMAY-G**: Rs. 1.30 lakh for pucca house (SC/ST priority)\n\n### Employment\n- **MGNREGS**: 100 days guaranteed work, Rs. 289/day\n\n### Water\n- **JJM**: Free tap water connection (tribal habitations prioritized)\n- **Basudha**: Piped water for uncovered habitations\n\n### Livelihood\n- **NRLM/OLM**: SHG formation, Rs. 15,000 revolving fund, bank loans\n- **Mission Shakti**: Interest-free loans up to Rs. 3 lakh\n\n### Infrastructure\n- **BGBO**: 40% allocation reserved for ITDA (tribal) blocks\n- **ORCP**: Road connectivity for habitations 125+ population (tribal areas)\n\n### Sanitation\n- **SBM-G**: Rs. 12,000 for toilet construction\n\nContact your GP office or Block Development Officer for application details.",
     "confidence_score": 0.85,
     "citizen_name": "Gurubari Hansda", "citizen_email": "gurubari.hansda@email.com",
     "citizen_key": "citizen3"},

    # 42 — Pending, infrastructure, medium, panchayati_raj related
    {"title": "Panchayat Bhawan construction stalled for 2 years in Nayagarh",
     "description": "Construction of a new Panchayat Bhawan was sanctioned 2 years ago under BGBO in our GP in Nayagarh. Foundation was laid but no further work done. The old Panchayat office is in a rented room. Rs. 15 lakh was sanctioned.",
     "district": "Nayagarh", "department": "infrastructure", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Raghunath Pradhan", "citizen_email": "raghunath.pradhan@email.com",
     "citizen_key": "citizen1"},

    # 43 — Pending, mgnregs, positive sentiment (appreciation + request)
    {"title": "Appreciate MGNREGS work quality — requesting extension to next village",
     "description": "The MGNREGS drainage channel built in our village in Khordha last month is excellent quality. It has solved our waterlogging problem completely. We request the same work to be extended to the adjacent habitation which still faces flooding.",
     "district": "Khordha", "department": "mgnregs", "priority": "low",
     "sentiment": "positive", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Prasanna Sahoo", "citizen_email": "prasanna.sahoo@email.com",
     "citizen_key": "citizen1"},

    # ======================================================================
    # NEW RESOLVED (13) — indices 44–56
    # ======================================================================

    # 44 — Hybrid, water supply, Cuttack, feedback 3 (average)
    {"title": "JJM tap connection leaking at meter — water wastage",
     "description": "Our JJM tap connection in Cuttack block has been leaking at the meter junction for 3 weeks. Water is being wasted continuously. We reported to the GP Jalasathi but the leak has not been fixed.",
     "district": "Cuttack", "department": "rural_water_supply", "priority": "medium",
     "sentiment": "negative", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "hybrid", "language": "english",
     "ai_resolution": "JJM tap connection leaks at the meter junction are typically caused by loose compression fittings or damaged washers. The GP Jalasathi should be able to fix this within 24 hours using standard repair kits provided under JJM O&M budget.",
     "manual_resolution": "RWSS JE visited site. Leak was caused by a cracked PVC elbow joint, not just a washer issue. Replacement fitting installed. However, the repair took 3 weeks due to material procurement delays. Jalasathi reprimanded for not escalating sooner.",
     "assigned_officer": "Er. Anil Panigrahi, EE-RWSS", "confidence_score": 0.55,
     "resolution_feedback": 3, "impact_score": 25,
     "notes": [
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Leak more severe than AI assessment indicated. Cracked PVC elbow, not a simple washer issue. Ordering replacement.", "note_type": "internal"},
     ],
     "citizen_name": "Sachidananda Mishra", "citizen_email": "sachida.mishra@email.com",
     "citizen_key": "citizen5"},

    # 45 — Officer manual, infrastructure, Jagatsinghpur, feedback 2 (poor)
    {"title": "BGBO road built last year already has potholes and cracks",
     "description": "A rural road constructed under BGBO scheme in Jagatsinghpur GP just 10 months ago has already developed deep potholes and surface cracks. The road cost Rs. 12 lakh but quality is very poor. We suspect the contractor used less bitumen.",
     "district": "Jagatsinghpur", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english",
     "manual_resolution": "Block JE conducted quality assessment and confirmed substandard construction. However, the contractor has already received full payment and the defect liability period clause was not enforced by the GP. Partial repair done under FC untied grants at additional cost of Rs. 3.5 lakh. Contractor issued warning letter but no penalty recovered.",
     "assigned_officer": "Sri Debashis Swain, Sr. District Officer", "resolution_feedback": 2,
     "impact_score": 50,
     "notes": [
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "Inspection confirms poor quality. Full payment already released — no leverage for penalty. Patching repair only option.", "note_type": "internal"},
     ],
     "citizen_name": "Akshay Nayak", "citizen_email": "akshay.nayak@email.com",
     "citizen_key": "citizen6"},

    # 46 — Officer manual, panchayati_raj, Bhadrak, feedback 1 (very poor)
    {"title": "GP Secretary refusing to issue certificates — demanding informal payments",
     "description": "The GP Secretary in our GP in Bhadrak is demanding Rs. 200-500 for issuing income certificates, residence certificates, and caste certificates. These should be free. When we complain, he threatens to delay our applications.",
     "district": "Bhadrak", "department": "panchayati_raj", "priority": "high",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english",
     "manual_resolution": "BDO investigated the complaint. GP Secretary denied the charges. Due to lack of recorded evidence (payments were in cash), only a verbal warning was issued. Citizens were advised to file RTI applications and record future interactions. The resolution was unsatisfactory as the GP Secretary continues in the same position.",
     "assigned_officer": "Smt. Priya Pattnaik, BDO", "resolution_feedback": 1,
     "impact_score": 62,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Investigated — GP Secretary denies charges. No documentary evidence from complainants. Verbal warning issued.", "note_type": "internal"},
     ],
     "citizen_name": "Nilambar Sethi", "citizen_email": "nilambar.sethi@email.com",
     "citizen_key": "citizen5"},

    # 47 — Hybrid, sanitation, Kendrapara, resolved with only internal notes
    {"title": "Community soak pit overflowing into village pathway",
     "description": "The community soak pit built under SBM-G in Kendrapara GP is overflowing. Grey water is flowing onto the main village pathway making it slippery and unhygienic. The pit seems undersized for the number of households connected.",
     "district": "Kendrapara", "department": "sanitation", "priority": "medium",
     "sentiment": "negative", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "hybrid", "language": "english",
     "ai_resolution": "Community soak pit overflow is commonly caused by undersized design, clogged filter layers, or excessive greywater inflow. Recommend Block SBM Coordinator to inspect pit dimensions versus connected households and arrange desludging if filter media is clogged.",
     "manual_resolution": "Block SBM Coordinator inspected. Soak pit designed for 20 households but 35 were connected. Filter gravel layer completely clogged. Desludging done immediately. Second soak pit sanctioned under SBM-G Phase II for the additional households. Completed within 30 days.",
     "assigned_officer": "Smt. Sarojini Das, Block Sanitation Coord.", "confidence_score": 0.70,
     "resolution_feedback": 4, "impact_score": 35,
     "notes": [
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "Pit capacity insufficient — 35 HH connected vs 20 HH design. Emergency desludging done. Second pit estimate prepared.", "note_type": "internal"},
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "Second soak pit construction order placed. Monitoring overflow daily until completion.", "note_type": "internal"},
     ],
     "citizen_name": "Trinath Barik", "citizen_email": "trinath.barik@email.com",
     "citizen_key": "citizen7"},

    # 48 — AI self-resolved, hindi, general dept
    {"title": "Gram Panchayat ka helpline number kya hai",
     "description": "Main Balasore mein rehta hoon. Mujhe Gram Panchayat se sambandhit shikayat karni hai lekin helpline number nahi pata. Kripya batayein ki kahan call karoon.",
     "district": "Balasore", "department": "general", "priority": "low",
     "sentiment": "neutral", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "hindi",
     "ai_resolution": "## Gram Panchayat Helpline aur Sampark Jaankari\n\n### Mukhya Helpline\n- **PR&DW Department Helpline**: 1800-345-6768 (toll-free)\n- **JJM/Water Supply Helpline**: 1916\n- **MGNREGS Helpline**: 1800-11-2233\n\n### Online Portal\n- **e-Gram Swaraj**: egramswaraj.gov.in\n- **Odisha Panchayat Portal**: panchayat.odisha.gov.in\n\n### Seedha Sampark\n- Apne **Block Development Officer (BDO)** se milein\n- BDO ka number panchayat.odisha.gov.in par mil jayega\n- District-wise BDO list bhi website par hai\n\n### Shikayat Darj Karne Ka Tarika\n1. GP office mein likhi shikayat dein\n2. Raseed zaroor lein\n3. 15 din mein jawab na mile toh BDO se sampark karein",
     "confidence_score": 0.92, "resolution_feedback": 5,
     "citizen_name": "Ramesh Patel", "citizen_email": None,
     "citizen_key": "citizen7"},

    # 49 — Officer manual, rural_livelihoods, Angul
    {"title": "SHG members denied bank loan despite good credit history",
     "description": "Our SHG 'Maa Mangala' in Angul has Grade-I rating and 3 years of regular savings. The bank branch manager rejected our loan application of Rs. 3 lakh without giving any reason. We have maintained 100% repayment on our previous Rs. 50,000 loan.",
     "district": "Angul", "department": "rural_livelihoods", "priority": "medium",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english",
     "manual_resolution": "Block OLM Coordinator escalated to Lead District Manager (LDM). Found bank branch had informally capped SHG lending due to NPA concerns from other groups. LDM directed the branch to process the loan per RBI-NRLM guidelines which mandate SHG lending. Loan of Rs. 3 lakh sanctioned within 15 days. Branch manager counseled on SHG lending norms.",
     "assigned_officer": "Sri Ranjit Mishra, DRDA PD", "resolution_feedback": 5,
     "impact_score": 40,
     "scheme_match": {"scheme_name": "NRLM / Odisha Livelihood Mission (OLM)", "relevance_score": 0.93,
                      "eligibility_likely": True, "eligibility_reasoning": "SHG with Grade-I rating seeking bank linkage loan under NRLM guidelines."},
     "citizen_name": "Suchitra Panda", "citizen_email": None,
     "citizen_key": "citizen6"},

    # 50 — Officer manual, mgnregs, Bargarh, odia language
    {"title": "MGNREGS re kaam chalichi kintu attendance fake achhi",
     "description": "Amara gaon Bargarh re MGNREGS re tala khodibar kaam chalichi. Kintu muster roll re jemane kaam karuchanti semane nku record karuchanti nahin, anya naama lekhuchanti. Fake attendance chaluchi. Dayakari tanta karantu.",
     "district": "Bargarh", "department": "mgnregs", "priority": "high",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "odia",
     "manual_resolution": "Block MGNREGS PO conducted surprise inspection at worksite. Confirmed discrepancies — 12 names on muster roll were not present at worksite. Aadhaar-based attendance verification initiated. Mate (supervisor) suspended pending inquiry. Corrected muster rolls prepared with actual workers. Wages for affected period recalculated and paid to genuine workers.",
     "assigned_officer": "Smt. Gitanjali Behera, MGNREGS APO", "resolution_feedback": 4,
     "impact_score": 58,
     "notes": [
         {"officer": "Smt. Gitanjali Behera, MGNREGS APO", "content": "Surprise inspection confirmed fake attendance. 12 ghost names found. Mate suspended. Aadhaar verification initiated for all active worksites.", "note_type": "internal"},
         {"officer": "Smt. Gitanjali Behera, MGNREGS APO", "content": "Fake attendance issue resolved. Mate suspended. Corrected muster rolls prepared. Your wages will be recalculated and credited within 7 days.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Gobinda Sahu", "citizen_email": None,
     "citizen_key": "citizen5"},

    # 51 — Hybrid, rural_housing, Puri, is_public
    {"title": "PMAY-G third installment delayed — house roof incomplete before monsoon",
     "description": "My PMAY-G house in Puri is at roof level but the third installment of Rs. 30,000 has not been released for 4 months. The roof is open and monsoon is approaching. I cannot complete the house without funds and my family has no shelter.",
     "district": "Puri", "department": "rural_housing", "priority": "urgent",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "hybrid", "language": "english", "is_public": True,
     "ai_resolution": "PMAY-G third installment delays at roof level are commonly caused by: (1) Awaas+ portal geo-tag rejection, (2) pending utilization certificate for second installment, or (3) State-level fund flow delay. Recommend checking Awaas+ portal status immediately.",
     "manual_resolution": "Block TA confirmed roof-level completion with fresh geo-tagged photos. Third installment was stuck due to a system glitch in Awaas+ portal batch processing. IT team at State PMAY-G cell resolved the technical issue. Third installment of Rs. 30,000 released within 10 days. Beneficiary completed roof before monsoon onset.",
     "assigned_officer": "Smt. Lopamudra Jena, BDO Housing", "confidence_score": 0.62,
     "resolution_feedback": 4, "impact_score": 70,
     "scheme_match": {"scheme_name": "PMAY-Gramin (Pradhan Mantri Awas Yojana - Gramin)", "relevance_score": 0.95,
                      "eligibility_likely": True, "eligibility_reasoning": "Beneficiary at roof-level stage of PMAY-G house, third installment pending."},
     "citizen_name": "Duryodhan Behera", "citizen_email": "duryodhan.behera@email.com",
     "citizen_key": "citizen5"},

    # 52 — Officer manual, infrastructure, Sambalpur, hindi
    {"title": "Gaon ka pul toot gaya hai — gaadi nahi ja sakti",
     "description": "Hamare gaon Sambalpur mein ek chhota pul barish mein toot gaya hai. Ab gaadi aur auto nahi ja sakti. Bachchon ko school jaane mein mushkil ho rahi hai. Hospital bhi door hai aur emergency mein koi ambulance nahi aa sakti.",
     "district": "Sambalpur", "department": "infrastructure", "priority": "urgent",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "hindi",
     "manual_resolution": "Block JE conducted emergency assessment. Found 6-meter span bridge washed out due to monsoon flash flood. Emergency measures: (1) Temporary steel plate crossing placed within 72 hours for light vehicles, (2) Permanent RCC culvert sanctioned under BGBO emergency provision — Rs. 12 lakh, (3) Construction completed within 45 days. Quality certificate issued after load testing.",
     "assigned_officer": "Sri Debashis Swain, Sr. District Officer", "resolution_feedback": 5,
     "impact_score": 78,
     "citizen_name": "Ramesh Patel", "citizen_email": None,
     "citizen_key": "citizen7"},

    # 53 — Hybrid, general, Mayurbhanj, positive sentiment
    {"title": "Excellent response to tribal welfare inquiry — thank you",
     "description": "I had filed a grievance asking about schemes for tribal households. The AI response was very comprehensive and helpful. I was able to apply for PMAY-G and MGNREGS job card based on the information. Thank you to the department.",
     "district": "Mayurbhanj", "department": "general", "priority": "low",
     "sentiment": "positive", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "hybrid", "language": "english",
     "ai_resolution": "Thank you for your positive feedback. We are glad the scheme information was helpful. For any future queries about tribal welfare schemes, you can also contact the Integrated Tribal Development Agency (ITDA) office in Mayurbhanj district.",
     "manual_resolution": "Citizen confirmed satisfaction with AI response. GP Secretary also provided physical copies of scheme application forms. Case closed with positive outcome.",
     "assigned_officer": "Sri Tapan Kumar Nayak, General Officer", "confidence_score": 0.88,
     "resolution_feedback": 5,
     "citizen_name": "Gurubari Hansda", "citizen_email": "gurubari.hansda@email.com",
     "citizen_key": "citizen3"},

    # 54 — Officer manual, sanitation, Ganjam
    {"title": "Faecal sludge from septic tanks being dumped in open fields",
     "description": "In our GP area in Ganjam, private tanker operators are dumping faecal sludge from septic tanks into agricultural fields at night. This is causing health hazards and contaminating the soil. The GP has no faecal sludge treatment plant.",
     "district": "Ganjam", "department": "sanitation", "priority": "high",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual", "language": "english",
     "manual_resolution": "Block SBM Coordinator initiated action: (1) Identified 3 private tanker operators through night surveillance, (2) Penalties imposed under Municipal Solid Waste Management Rules, (3) Emergency FSTP (Faecal Sludge Treatment Plant) sanctioned under SBM-G Phase II for the GP cluster — Rs. 28 lakh, (4) Until FSTP is operational, tankers directed to use the nearest municipal treatment facility (12 km away) with subsidy for transport cost.",
     "assigned_officer": "Smt. Sarojini Das, Block Sanitation Coord.", "resolution_feedback": 4,
     "impact_score": 65,
     "notes": [
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "Night surveillance identified 3 tanker operators. Penalties levied. FSTP proposal submitted under SBM-G Phase II.", "note_type": "internal"},
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "Illegal dumping stopped. Tankers redirected to municipal facility. New FSTP sanctioned for your GP cluster.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Nilambar Sethi", "citizen_email": "nilambar.sethi@email.com",
     "citizen_key": "citizen5"},

    # 55 — AI self-resolved, mgnregs, Deogarh, odia
    {"title": "MGNREGS re kaam pain demand kaise karibe",
     "description": "Mu MGNREGS re kaam chahe kintu GP office re kahinle demand likhita re deba kahile. Mu likhita demand kaise kariba se janinahin. Process kana se janaantu.",
     "district": "Deogarh", "department": "mgnregs", "priority": "low",
     "sentiment": "neutral", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "odia",
     "ai_resolution": "## MGNREGS re Kaam Pain Likhita Demand Prakriya\n\n### Step 1: Aavedan Lekhanti\n- Saadha kagaj re likhanti: \"Mu MGNREGS antargata kaam maaguchhi\"\n- **Job Card number** likhanti\n- **Tarikh** aur **dastaakhat** diantu\n\n### Step 2: GP Office re Jama Karantu\n- **GP Sarpanch** ba **GP Secretary** nku diantu\n- **Raseed** zaroor niantu\n\n### Step 3: Kaam Miliba\n- GP nku **15 din** bhitare kaam deba darkar\n- 15 din re na mile, **berojgaari bhatta** miliba\n\n### Mazdoori\n- Rs. **289/din** (Odisha, 2025-26)\n- Bank account re seedha jama heba",
     "confidence_score": 0.88, "resolution_feedback": 4,
     "citizen_name": "Trinath Barik", "citizen_email": "trinath.barik@email.com",
     "citizen_key": "citizen7"},

    # 56 — Hybrid, rural_water_supply, Kendujhar, scheme_match, feedback 3
    {"title": "Basudha bore well installed but yields very low water",
     "description": "A bore well was installed under Basudha scheme in our habitation in Kendujhar 2 months ago. But the yield is very low — only 2-3 households can get water before it runs dry. We are 80 families. The hydrogeological survey was not done properly.",
     "district": "Kendujhar", "department": "rural_water_supply", "priority": "high",
     "sentiment": "negative", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "hybrid", "language": "english",
     "ai_resolution": "Low bore well yield in Kendujhar district is often due to drilling into shallow aquifers in hard rock terrain. Recommend: (1) RWSS hydrogeological resurvey, (2) Possible deeper bore well or alternative surface water source, (3) Connection to nearest JJM multi-village scheme if available.",
     "manual_resolution": "RWSS hydrogeologist resurveyed the site. Original bore well drilled to only 60m in hard rock — yield test showed only 800 LPH (need 3,000 LPH for 80 families). Second bore well drilled at new location identified by geophysical survey — 120m depth, yield 4,500 LPH. Both connected to overhead tank with dual pump system. Supply now adequate.",
     "assigned_officer": "Er. Anil Panigrahi, EE-RWSS", "confidence_score": 0.60,
     "resolution_feedback": 3, "impact_score": 55,
     "scheme_match": {"scheme_name": "Basudha (Buxi Jagabandhu Assured Water Supply to Habitations)",
                      "relevance_score": 0.92,
                      "eligibility_likely": True, "eligibility_reasoning": "Habitation received Basudha bore well but with inadequate yield — requires remediation under same scheme."},
     "notes": [
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Original bore well yield inadequate. Hydrogeological resurvey ordered. Second drilling location identified.", "note_type": "internal"},
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Second bore well completed at 120m with adequate yield. Both wells connected to OHT. Water supply should now be sufficient for all 80 families.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Mamata Nayak", "citizen_email": "mamata.nayak@email.com",
     "citizen_key": "citizen4"},

    # ======================================================================
    # NEW IN PROGRESS (7) — indices 57–63
    # ======================================================================

    # 57 — In progress, sanitation, Balasore, SLA breached
    {"title": "SBM-G community toilet locked — no caretaker assigned",
     "description": "The community sanitary complex built under SBM-G in our market area in Balasore has been locked for 2 months. No caretaker has been assigned. The GP says there is no budget for a caretaker. People are using open defecation again.",
     "district": "Balasore", "department": "sanitation", "priority": "high",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Smt. Sarojini Das, Block Sanitation Coord.",
     "impact_score": 45,
     "sla_breached": True,
     "estimated_resolution_days": 20,
     "notes": [
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "GP does not have O&M budget for community toilet. Exploring SHG-based management model. Mission Shakti SHG 'Maa Durga' has expressed interest.", "note_type": "internal"},
     ],
     "citizen_name": "Suchitra Panda", "citizen_email": None,
     "citizen_key": "citizen6"},

    # 58 — In progress, rural_housing, Malkangiri, SLA breached, sub_tasks mixed
    {"title": "PMAY-G beneficiary list excludes eligible families — Gram Sabha not consulted",
     "description": "In our GP in Malkangiri, the PMAY-G beneficiary list was prepared without Gram Sabha approval. 15 eligible families from the permanent wait list were excluded. The Sarpanch included families that already have pucca houses. We demand re-verification.",
     "district": "Malkangiri", "department": "rural_housing", "priority": "high",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Smt. Lopamudra Jena, BDO Housing",
     "impact_score": 72,
     "sla_breached": True,
     "estimated_resolution_days": 25,
     "sub_tasks": [
         {"id": "st-mlk-1", "department": "panchayati_raj", "task": "Conduct special Gram Sabha for PMAY-G beneficiary list re-verification",
          "status": "resolved", "assigned_officer": "Smt. Priya Pattnaik, BDO"},
         {"id": "st-mlk-2", "department": "rural_housing", "task": "Cross-verify excluded families against SECC/Awaas+ permanent wait list and correct beneficiary list",
          "status": "in_progress", "assigned_officer": "Smt. Lopamudra Jena, BDO Housing"},
     ],
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Gram Sabha held. 15 excluded families confirmed eligible per SECC data. 4 included families found to own pucca houses — removed.", "note_type": "citizen_facing"},
         {"officer": "Smt. Lopamudra Jena, BDO Housing", "content": "Cross-verification with Awaas+ in progress. Corrected list to be uploaded within 10 days.", "note_type": "internal"},
     ],
     "citizen_name": "Padman Sabar", "citizen_email": "padman.sabar@email.com",
     "citizen_key": "citizen4"},

    # 59 — In progress, mgnregs, Nuapada, estimated overdue
    {"title": "MGNREGS drought-proofing work incomplete — monsoon approaching",
     "description": "Farm bunding and check dam work under MGNREGS was started in our village in Nuapada 3 months ago. Only 40% completed and work stopped. Workers have been demanding wages for completed portion. Monsoon will start in 6 weeks and incomplete structures will wash away.",
     "district": "Nuapada", "department": "mgnregs", "priority": "urgent",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Sri Bikram Sahu, MGNREGS PO",
     "impact_score": 65,
     "estimated_resolution_days": 10,
     "notes": [
         {"officer": "Sri Bikram Sahu, MGNREGS PO", "content": "Work stoppage due to mate dispute. New mate appointed. Additional workers mobilized. Target: complete before monsoon onset.", "note_type": "internal"},
         {"officer": "Sri Bikram Sahu, MGNREGS PO", "content": "Pending wages for completed portion being processed via FTO. New supervisor assigned. Work will resume within 3 days.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Ratan Majhi", "citizen_email": "ratan.majhi@email.com",
     "citizen_key": "citizen3"},

    # 60 — In progress, panchayati_raj, Subarnapur
    {"title": "Panchayat office has no internet — e-Gram Swaraj not functional",
     "description": "Our Panchayat office in Subarnapur has no internet connection for 6 months. The e-Gram Swaraj portal cannot be accessed. Fund utilization reports are not being uploaded. Citizens cannot track GP expenditure online. The GP says BSNL has not connected broadband.",
     "district": "Subarnapur", "department": "panchayati_raj", "priority": "medium",
     "sentiment": "negative", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Sri Hemanta Pradhan, GP Secretary",
     "estimated_resolution_days": 30,
     "notes": [
         {"officer": "Sri Hemanta Pradhan, GP Secretary", "content": "BSNL broadband complaint escalated to SDO. Alternate 4G dongle procured as temporary measure for e-Gram Swaraj access.", "note_type": "internal"},
     ],
     "citizen_name": "Niranjan Meher", "citizen_email": None,
     "citizen_key": "citizen1"},

    # 61 — In progress, infrastructure, Gajapati
    {"title": "BGBO bridge approach road washed out — village cut off during rains",
     "description": "The approach road to a small bridge in Gajapati GP built under BGBO last year has been washed out during recent heavy rains. The bridge itself is intact but vehicles cannot reach it. Our village of 200 families is cut off during every rain event.",
     "district": "Gajapati", "department": "infrastructure", "priority": "urgent",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english", "is_public": True,
     "assigned_officer": "Sri Debashis Swain, Sr. District Officer",
     "impact_score": 80,
     "estimated_resolution_days": 35,
     "notes": [
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "Bridge intact but both approach roads eroded. Gabion wall protection sanctioned. Rs. 5 lakh estimate under FC untied grants.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Kuni Sabar", "citizen_email": "kuni.sabar@email.com",
     "citizen_key": "citizen4"},

    # 62 — In progress, general, Jharsuguda, with citizen_phone
    {"title": "Cannot find information about government pension schemes for elderly",
     "description": "My father is 68 years old and lives in Jharsuguda village. He does not receive any pension. We have heard about Madhu Babu Pension Yojana and IGNOAPS but do not know how to apply. The GP office redirects us to Block office and Block office sends us back.",
     "district": "Jharsuguda", "department": "general", "priority": "low",
     "sentiment": "negative", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Sri Tapan Kumar Nayak, General Officer",
     "citizen_phone": "9876501234",
     "ai_resolution": "For elderly citizens in Odisha, the following pension schemes are available:\n\n1. **Madhu Babu Pension Yojana (MBPY)**: Rs. 500/month for BPL persons aged 60+\n2. **IGNOAPS**: Rs. 300/month (Central) + Rs. 200/month (State top-up) for BPL persons aged 60+\n\nApply at the GP office with: Aadhaar, age proof, BPL card, bank passbook. GP Secretary forwards to Block for approval.",
     "confidence_score": 0.75,
     "notes": [
         {"officer": "Sri Tapan Kumar Nayak, General Officer", "content": "Citizen redirected multiple times. Coordinating with Block office for direct application processing. Pension eligibility check initiated.", "note_type": "internal"},
     ],
     "citizen_name": "Trinath Barik", "citizen_email": "trinath.barik@email.com",
     "citizen_key": "citizen7"},

    # 63 — In progress, rural_water_supply, Boudh
    {"title": "JJM overhead tank leaking — water pressure too low for upper habitations",
     "description": "The JJM overhead tank built in Boudh block has developed cracks and is leaking from the base. Because of low water level, habitations on higher ground get no water while lower areas get excess flow. 120 families in upper areas affected.",
     "district": "Boudh", "department": "rural_water_supply", "priority": "high",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "language": "english",
     "assigned_officer": "Er. Anil Panigrahi, EE-RWSS",
     "impact_score": 68,
     "estimated_resolution_days": 25,
     "notes": [
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "OHT structural assessment shows cracks due to poor curing during construction. Epoxy grouting repair insufficient. Full tank relining required. Estimate Rs. 4.5 lakh under JJM O&M.", "note_type": "internal"},
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Tank repair work sanctioned. Temporary direct pumping arrangement to upper habitations until repair is complete (est. 3-4 weeks).", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Brundaban Sahu", "citizen_email": "brundaban.sahu@email.com",
     "citizen_key": "citizen1"},

    # ======================================================================
    # NEW ESCALATED (5) — indices 64–68
    # ======================================================================

    # 64 — Escalated, general, anonymous, corruption
    {"title": "Block office staff demanding bribes for processing MGNREGS and PMAY-G applications",
     "description": "At the Block Development Office in Nabarangpur, staff members are demanding Rs. 500-2000 to process MGNREGS job card applications and PMAY-G housing applications. Poor tribal families cannot pay and their applications are kept pending indefinitely. This is a systematic corruption issue.",
     "district": "Nabarangpur", "department": "general", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english",
     "impact_score": 88,
     "notes": [
         {"officer": "Sri Tapan Kumar Nayak, General Officer", "content": "Serious corruption allegation at Block level. Referred to District Collector and Anti-Corruption Bureau. All pending applications at the Block office being audited.", "note_type": "internal"},
     ],
     "citizen_name": None, "citizen_email": None, "is_anonymous": True,
     "citizen_key": "citizen3"},

    # 65 — Escalated, rural_livelihoods, scheme_match
    {"title": "Mission Shakti loan recovery agents harassing SHG members",
     "description": "Recovery agents from the bank are visiting our SHG members' homes in Kalahandi and threatening them about late loan repayment. Some members are only 15 days overdue due to crop failure. The agents are using abusive language and threatening to seize household items. This is not how Mission Shakti loans should work.",
     "district": "Kalahandi", "department": "rural_livelihoods", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english", "is_public": True,
     "impact_score": 75,
     "scheme_match": {"scheme_name": "Mission Shakti (SHG Convergence through PR&DW)", "relevance_score": 0.90,
                      "eligibility_likely": True, "eligibility_reasoning": "SHG formed under Mission Shakti facing harassment from bank recovery agents over loan repayment."},
     "notes": [
         {"officer": "Sri Ranjit Mishra, DRDA PD", "content": "Recovery agent harassment reported. Matter escalated to Lead District Manager and NABARD. RBI guidelines prohibit coercive recovery. Demand letter sent to bank branch.", "note_type": "internal"},
         {"officer": "Sri Ranjit Mishra, DRDA PD", "content": "We have escalated this to the Lead District Manager. Bank recovery agents are not allowed to use abusive language or threaten seizure of household items. You have rights under RBI fair practices code.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Sabita Nayak", "citizen_email": "sabita.nayak@email.com",
     "citizen_key": "citizen4"},

    # 66 — Escalated, sanitation, Nayagarh, high impact
    {"title": "Multiple villages report skin diseases after using contaminated well water",
     "description": "In 4 villages near Nayagarh town, villagers have been reporting skin rashes, itching, and boils after using well water. The wells are near an industrial effluent discharge point. Over 300 families affected. Children are worst hit. No alternative water source available.",
     "district": "Nayagarh", "department": "sanitation", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english", "is_public": True,
     "impact_score": 92,
     "notes": [
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "Health emergency. Water samples sent to State Pollution Control Board. Emergency tanker water supply arranged for 4 villages. District health team deployed for medical camps.", "note_type": "internal"},
         {"officer": "Smt. Sarojini Das, Block Sanitation Coord.", "content": "Emergency water tanker supply started for your village. Medical camp scheduled for next week. Water testing underway — results expected in 7 days.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Sushant Sahoo", "citizen_email": "sushant.sahoo@email.com",
     "citizen_key": "citizen1"},

    # 67 — Escalated, infrastructure, Angul, with assigned officer
    {"title": "BGBO school building construction collapsed during work — safety violation",
     "description": "A BGBO school building under construction in Angul GP partially collapsed yesterday. Fortunately no workers were inside at the time. The structure appears to have poor foundation work. This is the second infrastructure failure by the same contractor in our block.",
     "district": "Angul", "department": "infrastructure", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english",
     "impact_score": 85,
     "assigned_officer": "Sri Debashis Swain, Sr. District Officer",
     "notes": [
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "Structural collapse confirmed by Block JE. FIR being filed against contractor. Same contractor blacklisted in 2 other districts. Contract terminated. Fresh tender with mandatory structural engineer supervision.", "note_type": "internal"},
         {"officer": "Sri Debashis Swain, Sr. District Officer", "content": "The contractor responsible has been terminated and blacklisted. A fresh tender will be issued with mandatory structural engineer supervision. No workers were injured in the collapse.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Bijay Kumar Sahu", "citizen_email": "bijay.sahu@email.com",
     "citizen_key": "citizen2"},

    # 68 — Escalated, rural_water_supply, Dhenkanal
    {"title": "JJM pipeline cement mortar joints leaking at multiple villages — contractor quality issue",
     "description": "In 8 villages in Dhenkanal block, the JJM pipeline joints are leaking severely. The contractor used cement mortar instead of specified rubber ring joints. Water pressure drops to zero by the last 3 villages. Over 1,200 families affected. The same contractor is working in adjacent blocks.",
     "district": "Dhenkanal", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "language": "english",
     "impact_score": 93,
     "assigned_officer": "Er. Anil Panigrahi, EE-RWSS",
     "notes": [
         {"officer": "Er. Anil Panigrahi, EE-RWSS", "content": "Quality failure confirmed. Contractor used cement mortar joints (non-specification) for 8 km pipeline. Recall order issued for all 3 blocks where this contractor operates. Joint replacement work to begin immediately.", "note_type": "internal"},
     ],
     "citizen_name": "Surekha Pradhan", "citizen_email": "surekha.pradhan@email.com",
     "citizen_key": "citizen2",
     "systemic_link": True},

    # ======================================================================
    # NEW PENDING (22) — indices 69–90
    # ======================================================================

    # 69 — Pending, self_resolvable, english, rural_water_supply, citizen_phone
    {"title": "How to get water quality test done for our village bore well",
     "description": "We suspect our bore well water in Khordha has high TDS and may not be safe. Some people are getting stomach problems. How can we get the water tested? Is it free?",
     "district": "Khordha", "department": "rural_water_supply", "priority": "low",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "english",
     "citizen_phone": "9876512345",
     "ai_resolution": "## How to Get Water Quality Testing\n\n### Free Testing\n- **District RWSS Laboratory** provides free water quality testing\n- Collect sample in a clean 1-litre bottle, label with location\n- Submit at district RWSS lab or through GP Jalasathi\n\n### Parameters Tested\n- Bacteriological, pH, TDS, Fluoride, Arsenic, Iron, Nitrate\n- Results within **7 working days**\n\n### How to Request\n1. Contact your **GP Jalasathi** — they have Field Test Kits for basic testing\n2. For detailed lab testing, call **JJM Helpline 1916**\n3. Or visit the **District RWSS office** directly\n\n### If Contamination Found\n- Alternative supply arranged within **48 hours**\n- Long-term treatment plant installed under JJM/Basudha",
     "confidence_score": 0.90,
     "citizen_name": "Sachidananda Mishra", "citizen_email": "sachida.mishra@email.com",
     "citizen_key": "citizen5"},

    # 70 — Pending, self_resolvable, hindi, sanitation
    {"title": "SBM-G mein toilet banwane ka process kya hai",
     "description": "Main Mayurbhanj mein rehti hoon. Hamare ghar mein toilet nahi hai. SBM-G yojana ke baare mein suna hai ki sarkar Rs. 12,000 deti hai. Kya process hai aur kahan apply karoon?",
     "district": "Mayurbhanj", "department": "sanitation", "priority": "low",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "hindi",
     "citizen_phone": "9876523456",
     "ai_resolution": "## SBM-G Toilet Nirman Prakriya\n\n### Patrta (Eligibility)\n- Ghar mein toilet na ho\n- Gramin kshetra mein rehte hon\n- BPL/SC/ST parivaron ko pehle milta hai\n\n### Aavedan\n1. **GP office** mein jaayein\n2. GP Secretary se milein aur likhit aavedan dein\n3. **Aadhaar Card** aur **bank passbook** saath lein\n\n### Prakriya\n- GP Secretary patrta verify karega\n- Aap toilet banwayein (apni marzi ka design)\n- Photo evidence GP office mein dein\n- **Rs. 12,000** bank account mein jama hoga\n\n### Zaruri Baatein\n- Koi fee nahi lagti — puri tarah muft\n- Photo pehle aur baad mein — GP Secretary lega\n- Paisa toilet banne ke baad milega",
     "confidence_score": 0.91,
     "citizen_name": "Gurubari Hansda", "citizen_email": "gurubari.hansda@email.com",
     "citizen_key": "citizen3"},

    # 71 — Pending, self_resolvable, odia, rural_housing
    {"title": "PMAY-G re ghar baniba pain ki ki darkar heba",
     "description": "Mu Koraput re rahuchhi. Mu pucca ghar nahi. PMAY-G yojana re ghar banibaku chahuchhi kintu ki ki lagiba se januni. Application kaise kariba se janaantu.",
     "district": "Koraput", "department": "rural_housing", "priority": "low",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "self_resolvable",
     "resolution_type": "ai", "language": "odia",
     "ai_resolution": "## PMAY-G re Ghar Nirman Pain Prakriya\n\n### Patrta\n- Pucca ghar nahin thiba darkar\n- SECC 2011 list re naama thiba darkar\n- Awaas+ portal re check karantu\n\n### Darkar Document\n1. **Aadhaar Card**\n2. **Bank passbook**\n3. **SECC data** (GP office re check karantu)\n\n### Sahayata\n- Plain area: **Rs. 1,20,000**\n- Hilly/tribal area: **Rs. 1,30,000**\n- MGNREGS re 90 din kaam o miliba\n\n### Aavedan\n- GP office re jaantu\n- Gram Sabha re approval heba\n- 3 ta installment re paisa miliba\n- Geo-tagging photo darkar — foundation, lintel, completion",
     "confidence_score": 0.86,
     "scheme_match": {"scheme_name": "PMAY-Gramin (Pradhan Mantri Awas Yojana - Gramin)", "relevance_score": 0.94,
                      "eligibility_likely": True, "eligibility_reasoning": "Citizen without pucca house seeking PMAY-G assistance — direct scheme match."},
     "citizen_name": "Padman Sabar", "citizen_email": None,
     "citizen_key": "citizen4"},

    # 72 — Pending, is_public (0-vouch target), infrastructure, Cuttack
    {"title": "Street drain overflow flooding GP road during every rain",
     "description": "The main street drain in Cuttack GP overflows during every moderate rain, flooding the GP road. Shops and houses get waterlogged. The drain was last cleaned 2 years ago. Solid waste is blocking the drain at multiple points.",
     "district": "Cuttack", "department": "infrastructure", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english", "is_public": True,
     "impact_score": 40,
     "citizen_name": "Akshay Nayak", "citizen_email": "akshay.nayak@email.com",
     "citizen_key": "citizen6"},

    # 73 — Pending, rural_water_supply, Puri, citizen_phone
    {"title": "JJM tap providing water only 1 hour per day — insufficient for families",
     "description": "We received JJM tap connections in Puri block but water comes for only 1 hour in the morning. With 6 family members, we cannot fill enough water in 1 hour. Neighboring villages get 4-6 hours of supply. Why is our supply so limited?",
     "district": "Puri", "department": "rural_water_supply", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_phone": "9876534567",
     "citizen_name": "Duryodhan Behera", "citizen_email": "duryodhan.behera@email.com",
     "citizen_key": "citizen5"},

    # 74 — Pending, mgnregs, Ganjam, positive
    {"title": "Thank MGNREGS team for excellent pond deepening work — request more employment",
     "description": "The MGNREGS pond deepening work in our village in Ganjam was done very well. The pond now holds water throughout summer. I want to thank the team and request more MGNREGS work for land leveling in the adjacent area.",
     "district": "Ganjam", "department": "mgnregs", "priority": "low",
     "sentiment": "positive", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Gobinda Sahu", "citizen_email": None,
     "citizen_key": "citizen5"},

    # 75 — Pending, rural_livelihoods, Rayagada, scheme_match
    {"title": "SHG wants to start poultry business — need training and loan",
     "description": "Our SHG 'Maa Santoshi' in Rayagada has 15 members. We want to start a poultry farm as a group enterprise. We need skill training, initial loan, and guidance on where to sell. The Block OLM office has not responded to our request.",
     "district": "Rayagada", "department": "rural_livelihoods", "priority": "medium",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "scheme_match": {"scheme_name": "NRLM / Odisha Livelihood Mission (OLM)", "relevance_score": 0.85,
                      "eligibility_likely": True, "eligibility_reasoning": "SHG seeking livelihood enterprise support under NRLM/OLM for poultry farming."},
     "citizen_name": "Mamata Nayak", "citizen_email": "mamata.nayak@email.com",
     "citizen_key": "citizen4"},

    # 76 — Pending, infrastructure, Balangir, citizen_phone
    {"title": "BGBO community center built but no electricity connection",
     "description": "A community center was built under BGBO in our GP in Balangir 6 months ago. But there is still no electricity connection. The building cannot be used for meetings or community events in the evening. GP says the electricity connection was not in the project estimate.",
     "district": "Balangir", "department": "infrastructure", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_phone": "9876545678",
     "citizen_name": "Harihar Sahu", "citizen_email": "harihar.sahu@email.com",
     "citizen_key": "citizen1"},

    # 77 — Pending, is_public, sanitation, Subarnapur
    {"title": "GP declared ODF but 30% households have no toilet — false reporting",
     "description": "Our GP in Subarnapur was declared ODF (Open Defecation Free) but at least 30% of households still do not have toilets. The SBM verification was done superficially. We request a proper physical verification by the Block team.",
     "district": "Subarnapur", "department": "sanitation", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "language": "english", "is_public": True,
     "impact_score": 55,
     "citizen_name": "Niranjan Meher", "citizen_email": None,
     "citizen_key": "citizen1"},

    # 78 — Pending, rural_housing, Sundargarh, is_anonymous
    {"title": "Contractor for PMAY-G houses using single brick walls instead of double",
     "description": "The contractor hired for PMAY-G house construction in several villages in Sundargarh is building single-brick walls instead of the required double-brick walls. The houses will not be structurally sound. Some villagers have complained but the contractor is threatening them.",
     "district": "Sundargarh", "department": "rural_housing", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "escalation",
     "language": "english", "is_anonymous": True,
     "impact_score": 70,
     "citizen_name": None, "citizen_email": None,
     "citizen_key": "citizen5"},

    # 79 — Pending, mgnregs, Kandhamal, scheme_match
    {"title": "MGNREGS convergence with PMAY-G — 90 days labor component not provided",
     "description": "I am building my PMAY-G house in Kandhamal. Under the scheme, I should get 90 person-days of unskilled labor under MGNREGS convergence. But the GP says MGNREGS and PMAY-G are separate schemes and refuses to issue work days. I need this labor support to complete my house.",
     "district": "Kandhamal", "department": "mgnregs", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "scheme_match": {"scheme_name": "MGNREGS (Mahatma Gandhi National Rural Employment Guarantee Scheme)", "relevance_score": 0.88,
                      "eligibility_likely": True, "eligibility_reasoning": "PMAY-G beneficiary entitled to 90 person-days MGNREGS convergence for house construction."},
     "citizen_name": "Gobinda Kanhar", "citizen_email": "gobinda.kanhar@email.com",
     "citizen_key": "citizen3"},

    # 80 — Pending, panchayati_raj, Kalahandi, is_public
    {"title": "GP not displaying expenditure details on notice board as mandated",
     "description": "Our GP in Kalahandi is not displaying expenditure details, beneficiary lists, or project details on the Panchayat notice board as required by the Odisha Panchayat Raj Act. Citizens have no transparency on how GP funds are being used.",
     "district": "Kalahandi", "department": "panchayati_raj", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english", "is_public": True,
     "impact_score": 35,
     "citizen_name": "Laxman Naik", "citizen_email": "laxman.naik@email.com",
     "citizen_key": "citizen3"},

    # 81 — Pending, rural_water_supply, Malkangiri, no AI resolution (freshly filed)
    {"title": "Hand pump installed 6 months ago already broken — no spare parts available",
     "description": "The India Mark II hand pump installed in our tribal village in Malkangiri has broken down. The connecting rod snapped. The GP says spare parts are not available in the district. Women are walking 4 km to fetch water from a stream.",
     "district": "Malkangiri", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "impact_score": 78,
     "citizen_name": "Padman Sabar", "citizen_email": "padman.sabar@email.com",
     "citizen_key": "citizen4"},

    # 82 — Pending, infrastructure, Dhenkanal, positive
    {"title": "Appreciate new BGBO market shed — request for drainage around it",
     "description": "The new market shed built under BGBO in Dhenkanal GP is very well constructed and has benefited many vendors. However, during rains, water accumulates around the shed because there is no drainage. Request construction of a small drain to make it fully functional.",
     "district": "Dhenkanal", "department": "infrastructure", "priority": "low",
     "sentiment": "positive", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Surekha Pradhan", "citizen_email": "surekha.pradhan@email.com",
     "citizen_key": "citizen2"},

    # 83 — Pending, rural_livelihoods, Boudh
    {"title": "SHG bank account frozen by bank without notice",
     "description": "Our SHG 'Maa Mangala' in Boudh has been saving regularly for 18 months. The bank suddenly froze our joint account without any notice. We cannot deposit savings or withdraw funds. The bank says it is due to KYC non-compliance but we submitted all documents.",
     "district": "Boudh", "department": "rural_livelihoods", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Suchitra Panda", "citizen_email": None,
     "citizen_key": "citizen6"},

    # 84 — Pending, sanitation, Jajpur, citizen_phone
    {"title": "Plastic waste burning in open near school — children inhaling toxic smoke",
     "description": "Near the GP school in Jajpur, villagers burn plastic waste in an open pit daily. The toxic smoke enters the school during morning hours when wind blows towards the school. Children are coughing and getting respiratory problems. The GP has no waste segregation system.",
     "district": "Jajpur", "department": "sanitation", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_phone": "9876556789",
     "impact_score": 60,
     "citizen_name": "Lalita Sahoo", "citizen_email": "lalita.sahoo@email.com",
     "citizen_key": "citizen2"},

    # 85 — Pending, panchayati_raj, Deogarh, hindi
    {"title": "GP Secretary school certificate ke liye Rs. 500 maang raha hai",
     "description": "Hamare GP Secretary Deogarh mein har certificate ke liye Rs. 500 maangta hai. Income certificate, residence certificate, sab ke liye paisa lagta hai. Ye galat hai. Humlog garib hain aur ek ek paisa mushkil se kamaate hain.",
     "district": "Deogarh", "department": "panchayati_raj", "priority": "medium",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "language": "hindi",
     "citizen_name": "Sukanta Mahapatra", "citizen_email": None,
     "citizen_key": "citizen1"},

    # 86 — Pending, general, Kendrapara, is_anonymous
    {"title": "Block office staff not present during office hours — no one to help citizens",
     "description": "The Block Development Office in Kendrapara is frequently empty during office hours. Staff come late, leave early, and take long lunch breaks. When citizens visit for work, they are told to come another day. This happens regularly.",
     "district": "Kendrapara", "department": "general", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english", "is_anonymous": True,
     "citizen_name": None, "citizen_email": None,
     "citizen_key": "citizen2"},

    # 87 — Pending, rural_water_supply, Nabarangpur
    {"title": "Solar pump for water supply not working — no maintenance team",
     "description": "The solar-powered water pump installed under JJM in our tribal village in Nabarangpur stopped working 3 weeks ago. The solar panels seem dirty and the pump motor makes strange noises. There is no maintenance team or trained person in the village.",
     "district": "Nabarangpur", "department": "rural_water_supply", "priority": "high",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "impact_score": 55,
     "scheme_match": {"scheme_name": "Jal Jeevan Mission (JJM)", "relevance_score": 0.89,
                      "eligibility_likely": True, "eligibility_reasoning": "JJM solar pump installation requiring maintenance under scheme O&M provisions."},
     "citizen_name": "Sabita Nayak", "citizen_email": "sabita.nayak@email.com",
     "citizen_key": "citizen4"},

    # 88 — Pending, mgnregs, Angul
    {"title": "MGNREGS plantation work — saplings died due to no follow-up watering",
     "description": "Under MGNREGS, 500 saplings were planted along the village road in Angul 3 months ago. But no watering or maintenance work was provided afterwards. 80% of saplings have died. This is a waste of public money. We demand replanting with proper maintenance plan.",
     "district": "Angul", "department": "mgnregs", "priority": "low",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "citizen_name": "Bijay Kumar Sahu", "citizen_email": "bijay.sahu@email.com",
     "citizen_key": "citizen2"},

    # 89 — Pending, infrastructure, Mayurbhanj, is_public
    {"title": "BGBO road to weekly market unusable — tribal families losing income",
     "description": "The road connecting 5 tribal villages to the weekly market in Mayurbhanj is in terrible condition. Deep potholes, no shoulder, completely impassable in monsoon. Tribal families cannot transport forest produce to market and are losing income. We requested BGBO repair 8 months ago.",
     "district": "Mayurbhanj", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "language": "english", "is_public": True,
     "impact_score": 72,
     "citizen_name": "Gurubari Hansda", "citizen_email": "gurubari.hansda@email.com",
     "citizen_key": "citizen3"},

    # 90 — Pending, rural_housing, Gajapati, positive + scheme_match
    {"title": "PMAY-G house completed successfully — thank you for the support",
     "description": "I want to thank the department for the PMAY-G housing assistance. My family now has a pucca house for the first time. The Block TA was very helpful with the geo-tagging process. I request that my neighbor's family also be included in the next beneficiary list as they still live in a kutcha house.",
     "district": "Gajapati", "department": "rural_housing", "priority": "low",
     "sentiment": "positive", "status": "pending", "resolution_tier": "officer_action",
     "language": "english",
     "scheme_match": {"scheme_name": "PMAY-Gramin (Pradhan Mantri Awas Yojana - Gramin)", "relevance_score": 0.85,
                      "eligibility_likely": True, "eligibility_reasoning": "Citizen expressing gratitude for PMAY-G and requesting inclusion of neighbor in next beneficiary list."},
     "citizen_name": "Kuni Sabar", "citizen_email": "kuni.sabar@email.com",
     "citizen_key": "citizen4"},
]


# ---------------------------------------------------------------------------
# Import function
# ---------------------------------------------------------------------------
_ATTACHMENT_INDICES = {3, 12, 20, 25, 50, 55, 60, 65, 70, 75, 80}

# Spread grievances from 6 months ago to today for charts and reports
SIX_MONTHS_DAYS = 183

async def import_grievances(db, user_ids: dict[str, str], *, file_ids: dict | None = None) -> list[dict]:
    """Insert all seed grievances. Returns list of inserted docs (for extras to reference)."""
    print("\n  Importing grievances...")
    now = now_utc()
    inserted: list[dict] = []
    attachment_pool = (file_ids or {}).get("grievance_attachments", [])
    n_total = len(GRIEVANCES)

    for i, g in enumerate(GRIEVANCES):
        # Tracking number via counter
        seq = db.counters.find_one_and_update(
            {"_id": "grievance"}, {"$inc": {"seq": 1}},
            upsert=True, return_document=True)
        tracking = f"GRV-{now.year}-{seq['seq']:06d}"

        is_anon = g.get("is_anonymous", False)
        status  = g["status"]
        dept    = g["department"]
        pri     = g["priority"]
        sla_hrs = PRIORITY_SLA_HOURS.get(pri, 168)

        # Timestamps: spread evenly from 6 months ago to today
        if n_total <= 1:
            days_ago = SIX_MONTHS_DAYS // 2
        else:
            days_ago = SIX_MONTHS_DAYS - (i * SIX_MONTHS_DAYS / (n_total - 1))
        created = now - timedelta(days=days_ago, hours=i % 24)

        if status == "resolved":
            updated = created + timedelta(days=3, hours=i * 2)
            sla     = created + timedelta(hours=sla_hrs)
        elif status == "in_progress":
            updated = now - timedelta(days=2, hours=i)
            sla     = (now - timedelta(hours=6)) if g.get("sla_breached") else (now + timedelta(hours=sla_hrs))
        elif status == "escalated":
            updated = now - timedelta(days=1)
            sla     = now + timedelta(hours=12)
        else:  # pending
            updated = created + timedelta(hours=i)
            sla     = (now - timedelta(hours=12)) if g.get("sla_breached") else (now + timedelta(hours=sla_hrs))

        # Estimated resolution deadline (for in_progress entries)
        est_deadline = None
        if g.get("estimated_resolution_days"):
            est_deadline = created + timedelta(days=g["estimated_resolution_days"])

        # Notes with timestamps (relative to grievance creation, within a few days)
        seed_notes = []
        for n in g.get("notes", []):
            seed_notes.append({
                "officer": n["officer"],
                "content": n["content"],
                "note_type": n["note_type"],
                "created_at": created + timedelta(days=1, hours=len(seed_notes) * 6),
            })

        # Citizen user linkage
        citizen_key = g.get("citizen_key", "citizen1")
        citizen_uid = user_ids.get(citizen_key)

        doc = {
            "_id": new_id(),
            "tracking_number": tracking,
            "title": g["title"],
            "description": g["description"],
            "citizen_name": None if is_anon else g.get("citizen_name"),
            "citizen_email": None if is_anon else g.get("citizen_email"),
            "citizen_phone": None if is_anon else g.get("citizen_phone"),
            "is_anonymous": is_anon,
            "is_public": g.get("is_public", False) and not is_anon,
            "language": g.get("language", "english"),
            "district": g.get("district"),
            "department": dept,
            "priority": pri,
            "officer_category": OFFICER_CATS.get(dept, "general_officer"),
            "status": status,
            "sentiment": g.get("sentiment", "neutral"),
            "created_at": created,
            "updated_at": updated,
            "sla_deadline": sla,
            "estimated_resolution_deadline": est_deadline,
            "ai_resolution": g.get("ai_resolution"),
            "manual_resolution": g.get("manual_resolution"),
            "resolution_tier": g.get("resolution_tier", "officer_action"),
            "resolution_type": g.get("resolution_type"),
            "confidence_score": g.get("confidence_score", 0.0),
            "assigned_officer": g.get("assigned_officer"),
            "resolution_feedback": g.get("resolution_feedback"),
            "notes": seed_notes,
            "location": geojson_point(g["district"]) if g.get("district") else None,
            "citizen_user_id": citizen_uid,
            "attachments": (
                [attachment_pool[i % len(attachment_pool)]]
                if attachment_pool and i in _ATTACHMENT_INDICES
                else []
            ),
            "impact_score": g.get("impact_score"),
            "scheme_match": g.get("scheme_match"),
            "sub_tasks": g.get("sub_tasks", []),
            "systemic_issue_id": None,          # filled later by extras
        }
        db.grievances.insert_one(doc)
        inserted.append(doc)

        tag = {"resolved": "OK", "in_progress": "WIP", "escalated": "ESC", "pending": "NEW"}.get(status, "")
        print(f"    [{i+1:2d}/{len(GRIEVANCES)}] {tag:3s}  {tracking}  {g['title'][:52]}...")

    # Indexes
    db.grievances.create_index("created_at")
    db.grievances.create_index("status")
    db.grievances.create_index("department")
    db.grievances.create_index("priority")
    db.grievances.create_index("tracking_number")
    db.grievances.create_index("citizen_user_id")
    db.grievances.create_index("is_public")

    print(f"  => {len(GRIEVANCES)} grievances imported")
    return inserted
