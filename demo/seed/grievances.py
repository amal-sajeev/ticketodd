# Seed data: Grievances (~45 entries exercising every feature)
#
# Coverage matrix:
#   Statuses   : pending (18), in_progress (8), resolved (12), escalated (5)
#   Departments: all 8 represented (5-7 each)
#   Priorities : low, medium, high, urgent
#   Sentiments : positive, neutral, negative, frustrated
#   Res. tiers : self_resolvable, officer_action, escalation
#   Res. types : ai, manual, hybrid
#   Special    : is_public, is_anonymous, scheme_match, sub_tasks,
#                impact_score, estimated_resolution_deadline, notes,
#                multiple languages, location on all entries

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
                      "reasoning": "Grievance directly relates to SBM-G IHHL incentive payment delay."},
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
                      "reasoning": "Citizen is asking about SBM-G toilet construction documents — direct scheme match."},
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
                      "reasoning": "Direct application under Nirman Shramik housing scheme for construction workers."},
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
                      "reasoning": "Grievance is about delayed PMAY-G installment release at lintel stage."},
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
                      "reasoning": "Citizen is requesting bore well under Basudha scheme. Direct scheme match for water supply to unserved habitation."},
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
]


# ---------------------------------------------------------------------------
# Import function
# ---------------------------------------------------------------------------
_ATTACHMENT_INDICES = {3, 12, 20, 25}

async def import_grievances(db, user_ids: dict[str, str], *, file_ids: dict | None = None) -> list[dict]:
    """Insert all seed grievances. Returns list of inserted docs (for extras to reference)."""
    print("\n  Importing grievances...")
    now = now_utc()
    inserted: list[dict] = []
    attachment_pool = (file_ids or {}).get("grievance_attachments", [])

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

        # Timestamps
        created = now - timedelta(days=42 - i)   # spread so newest land within this week
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

        # Notes with timestamps
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
            "citizen_phone": None,
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
