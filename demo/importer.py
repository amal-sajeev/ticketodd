# Panchayati Raj & Drinking Water Department — Seed Data Importer
# Populates MongoDB + Qdrant with PR&DW-specific demo data

import os
import uuid
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import AsyncOpenAI
from passlib.context import CryptContext
from dotenv import load_dotenv

# Load .env from same directory, one level up, or cwd
_script_dir = Path(__file__).resolve().parent
for _env_path in [_script_dir / ".env", _script_dir.parent / ".env", Path.cwd() / ".env"]:
    if _env_path.is_file():
        load_dotenv(_env_path, override=True)
        break
else:
    load_dotenv(override=True)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = 3072

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -------------------------------------------------------------------
# Seed Data: Documentation
# -------------------------------------------------------------------
DOCUMENTATION = [
    {"title": "Jal Jeevan Mission — Tap Water Connection Process",
     "content": "Under Jal Jeevan Mission (JJM), every rural household is entitled to a Functional Household Tap Connection (FHTC). Apply at the Gram Panchayat office or through the Block Development Officer. Required: Aadhaar, household ID, BPL certificate (for priority). The RWSS division executes the pipeline work. Timeline: New connections within 30-60 days of approved DPR. Report non-functional taps to the GP Jalasathi or call the JJM helpline 1916.",
     "category": "rural_water_supply"},
    {"title": "Basudha Piped Water Supply Scheme",
     "content": "Buxi Jagabandhu Assured Water Supply to Habitations (BASUDHA) provides piped drinking water to habitations not covered under JJM. Funded by the state government. Report pipeline leaks, low pressure, or contamination to the RWSS Executive Engineer at the block level. Emergency repairs within 48 hours. Water quality testing available free at district RWSS labs. Helpline: 1916.",
     "category": "rural_water_supply"},
    {"title": "MGNREGS Job Card Application and Wage Complaint",
     "content": "Every rural household is entitled to a MGNREGS job card. Apply at the Gram Panchayat with passport-size photo and Aadhaar. Job card is issued within 15 days. Demand work by written application to GP Sarpanch. Wages must be paid within 15 days of work completion via direct bank transfer. If wages are delayed, complain to the Block MGNREGS Programme Officer or the District Programme Coordinator. Wage rate: Rs. 289/day (Odisha, 2025-26).",
     "category": "mgnregs"},
    {"title": "BGBO Scheme — Rural Infrastructure Projects",
     "content": "Bikashita Gaon Bikashita Odisha (BGBO) is the state flagship rural infrastructure scheme with Rs. 1,000 crore annual allocation. Covers roads, bridges, culverts, school buildings, civic amenities, and sports facilities. Projects are proposed at Gram Sabha and approved by Block/District. Maximum 35% funds for roads. 40% allocation reserved for ITDA blocks. Minimum project estimate Rs. 3 lakh. Report project quality issues to BDO or District Panchayat Officer.",
     "category": "infrastructure"},
    {"title": "PMAY-Gramin — Rural Housing Application",
     "content": "Pradhan Mantri Awas Yojana - Gramin provides financial assistance for pucca house construction. Assistance: Rs. 1.20 lakh (plain areas), Rs. 1.30 lakh (hilly/difficult areas) plus MGNREGS convergence for 90/95 person-days of unskilled labour. Eligibility: Households with no pucca house as per SECC 2011 data. Apply at GP office. Geo-tagging required at foundation, lintel, and completion stages for installment release. Track at awaassoft.nic.in.",
     "category": "rural_housing"},
    {"title": "Gram Sabha and Palli Sabha Meeting Process",
     "content": "Gram Sabha meetings must be held at least twice a year (2nd October and 26th January). Palli Sabha (ward-level meeting) is held quarterly. Agenda includes: approval of beneficiary lists, review of GP works, discussion of grievances, MGNREGS work demands. Quorum: 1/10th of voters. Citizens can demand a special Gram Sabha through written request signed by 1/3rd voters. Complaints about non-conduct of meetings go to the BDO.",
     "category": "panchayati_raj"},
    {"title": "Swachh Bharat Mission - Gramin (SBM-G) Toilet Construction",
     "content": "Under SBM-G, eligible households without toilets receive Rs. 12,000 incentive for constructing Individual Household Latrines (IHHL). Apply at GP office. Verification by GP Secretary and Block Coordinator. Payment after construction and verification via photo evidence. Solid and Liquid Waste Management (SLWM) projects at village level. Report ODF slippage or SLWM issues to Block SBM Coordinator.",
     "category": "sanitation"},
    {"title": "NRLM / Odisha Livelihood Mission — SHG Formation",
     "content": "National Rural Livelihoods Mission (NRLM) and Odisha Livelihood Mission (OLM) support formation of Self Help Groups, bank linkage, and enterprise development. Women from BPL families form groups of 10-20 members. SHGs receive Revolving Fund (Rs. 15,000) and Community Investment Fund. Bank linkage loans up to Rs. 3 lakh at reduced interest. Contact Block OLM Coordinator or Community Resource Person at GP level.",
     "category": "rural_livelihoods"},
    {"title": "Bore Well Repair and New Installation",
     "content": "Bore well complaints (non-functional pump, low yield, water quality) should be reported to the GP Jalasathi or RWSS Junior Engineer at the Block level. Repair timeline: 72 hours for emergency, 15 days for routine. New bore well requests are submitted through GP resolution to Block RWSS. Installation under JJM or Basudha based on habitation priority. Water quality testing mandatory before commissioning.",
     "category": "rural_water_supply"},
    {"title": "Water Quality Testing and Contamination Complaint",
     "content": "Citizens can request free water quality testing at district RWSS laboratories. Report suspected contamination (colour, odour, illness) to the GP Jalasathi or call helpline 1916 for immediate response. Parameters tested: bacteriological, chemical (fluoride, arsenic, iron, nitrate). Results within 7 days. If contamination confirmed, alternative supply arranged within 48 hours. Long-term treatment plant installed under JJM.",
     "category": "rural_water_supply"},
    {"title": "Finance Commission Grants — Utilization and Query",
     "content": "15th Finance Commission grants are released to GPs, Blocks, and Zilla Parishads for basic services: drinking water, sanitation, solid waste management, roads, street lighting, and local governance. Tied grants (50%) must be used for water/sanitation. Untied grants for any GP-approved priority. Citizens can query fund utilization at Gram Sabha or through the e-Gram Swaraj portal (egramswaraj.gov.in). Complaints about misuse to BDO or District Panchayat Officer.",
     "category": "infrastructure"},
    {"title": "Sarpanch and GP Grievance Escalation",
     "content": "If grievances are not resolved at GP level, citizens can escalate to the Block Development Officer (BDO). BDO must acknowledge within 3 days and resolve within 15-30 days. Further escalation to District Panchayat Officer or District Collector. Written complaints preferred with acknowledgment receipt. Panchayat.odisha.gov.in portal provides contact details for all BDOs and District Officers.",
     "category": "panchayati_raj"},
    {"title": "Nirman Shramik Pucca Ghar Yojana — Worker Housing",
     "content": "Registered construction workers can apply for housing assistance under Nirman Shramik Pucca Ghar Yojana through the Panchayati Raj Department. Eligibility: Registered with Odisha Building and Construction Workers Welfare Board, no existing pucca house. Financial assistance for house construction. Apply at GP or Block office with registration certificate, Aadhaar, and land documents.",
     "category": "rural_housing"},
    {"title": "Rural Road Repair Under BGBO / Finance Commission",
     "content": "Rural road repair complaints (potholes, damaged culverts, washed-out sections) should be reported to GP Secretary or Block technical staff. Emergency repairs prioritized during monsoon. BGBO funds can be used for roads up to 35% of allocation. Finance Commission grants also cover road maintenance. GP must pass a resolution for major repairs. Quality inspection by Block Junior Engineer. Citizens can attend Gram Sabha to demand road projects.",
     "category": "infrastructure"},
    {"title": "Panchayat Election and Representative Information",
     "content": "Panchayat elections in Odisha are conducted at three tiers: Gram Panchayat (Sarpanch, Ward Members), Panchayat Samiti (Block level), and Zilla Parishad (District level). Elections held every 5 years. List of elected representatives available at panchayat.odisha.gov.in. To file complaints about representatives (non-performance, misconduct), approach the BDO or State Election Commission. No-confidence motion requires 2/3rd majority of GP members.",
     "category": "panchayati_raj"},
]

# -------------------------------------------------------------------
# Seed Data: Service Memory (Past resolved grievances)
# -------------------------------------------------------------------
SERVICE_MEMORY = [
    {"query": "Bore well not working in tribal village Koraput for 3 months",
     "resolution": "RWSS Junior Engineer inspected the site. Found submersible pump motor burned out. Replacement motor installed under Basudha maintenance budget within 5 days. GP Jalasathi trained on basic maintenance. Quarterly preventive check schedule established.",
     "category": "rural_water_supply", "agent_name": "EE-RWSS Panigrahi"},
    {"query": "MGNREGS wages not paid for 45 days in Kalahandi block",
     "resolution": "Block MGNREGS Programme Officer investigated. Found FTO (Fund Transfer Order) stuck due to incorrect bank account details for 23 workers. Accounts corrected via Aadhaar re-verification. Wages for 45 days (Rs. 13,005 per worker) transferred within 7 days. Delay compensation of Rs. 0.05% per day also credited.",
     "category": "mgnregs", "agent_name": "BPO Sahu"},
    {"query": "PMAY-G second installment not released after completing lintel in Ganjam",
     "resolution": "Block Technical Assistant visited site and confirmed lintel-level completion. Found geo-tagging photo was blurry and rejected by system. Re-photographed with proper GPS tagging. Second installment of Rs. 40,000 released within 10 days via Awaas+ portal.",
     "category": "rural_housing", "agent_name": "BDO Jena"},
    {"query": "BGBO road project incomplete and abandoned in Rayagada",
     "resolution": "District Panchayat Officer conducted field inspection. Contractor had abandoned work at 60% completion citing material cost escalation. Contract terminated and penalty imposed. New agency engaged through re-tendering. Road completed within 45 days. Quality certificate issued by Block JE.",
     "category": "infrastructure", "agent_name": "DPO Mishra"},
    {"query": "No JJM tap water connection despite being in DPR in Mayurbhanj",
     "resolution": "Verified that habitation was included in approved District Plan for JJM. Pipeline laying was delayed due to rocky terrain. RWSS deployed percussion drilling team. Pipeline extended to the habitation and 47 household connections provided within 30 days. Chlorination unit installed.",
     "category": "rural_water_supply", "agent_name": "EE-RWSS Mohapatra"},
    {"query": "SBM-G toilet subsidy not received after construction in Balasore",
     "resolution": "Block SBM Coordinator verified construction through physical inspection and photo evidence. Found that beneficiary was not registered in IMIS portal. Entry completed by GP Secretary. Rs. 12,000 incentive credited to beneficiary bank account within 15 days.",
     "category": "sanitation", "agent_name": "Block Coordinator Das"},
    {"query": "NRLM SHG bank linkage loan pending for 6 months in Nabarangpur",
     "resolution": "Block OLM Coordinator intervened with the bank branch manager. Found SHG credit rating was pending due to missing monthly meeting records. OLM CRP helped SHG complete documentation. Bank loan of Rs. 2 lakh sanctioned within 15 days of resubmission.",
     "category": "rural_livelihoods", "agent_name": "Block OLM Coordinator Majhi"},
    {"query": "Water quality complaint — yellow water from JJM tap in Dhenkanal",
     "resolution": "RWSS collected water samples from 5 points in the pipeline network. Lab results showed high iron content (3.2 mg/L vs 0.3 mg/L standard). Iron removal plant ordered and installed within 20 days under JJM water quality component. Follow-up testing confirmed levels within safe limits.",
     "category": "rural_water_supply", "agent_name": "EE-RWSS Behera"},
    {"query": "Gram Panchayat not holding Gram Sabha for over a year in Boudh",
     "resolution": "BDO issued notice to Sarpanch under Odisha Gram Panchayat Act. Show-cause notice served within 7 days. Gram Sabha scheduled within 15 days with BDO personally supervising. Ward members notified through public notice. Gram Sabha held with quorum met. Regular schedule enforced going forward.",
     "category": "panchayati_raj", "agent_name": "BDO Pradhan"},
    {"query": "MGNREGS job card not issued despite application 2 months ago in Nuapada",
     "resolution": "GP Secretary had lost the application. Duplicate application accepted immediately. Job card with household photo issued within 7 days. 5 days of work allocated in ongoing village road project. GP Secretary warned for negligence. Digital application process recommended.",
     "category": "mgnregs", "agent_name": "BPO Nayak"},
    {"query": "Finance Commission grant misuse alleged at GP level in Kendrapara",
     "resolution": "District Panchayat Officer ordered audit of GP accounts. Found irregular expenditure of Rs. 3.5 lakh on unauthorized items. Sarpanch issued show-cause notice. Amount ordered to be recovered. Utilization certificate rejected. Matter referred to Vigilance for further inquiry. Untied grant release paused pending compliance.",
     "category": "infrastructure", "agent_name": "DPO Swain"},
    {"query": "BGBO community hall construction — poor quality material used in Sundargarh",
     "resolution": "Block JE conducted quality assessment. Concrete cube test failed — strength below specification. Contractor directed to demolish and rebuild the affected portion at own cost. Penalty of 10% deducted from payment. Third-party quality monitoring engaged for remaining work.",
     "category": "infrastructure", "agent_name": "BDO Agrawal"},
    {"query": "Mission Shakti SHG not receiving revolving fund in Gajapati",
     "resolution": "Block OLM Coordinator found SHG was registered but not graded. CRP conducted grading exercise. SHG qualified for Grade-I rating. Revolving Fund of Rs. 15,000 disbursed within 10 days through NRLM fund flow. SHG linked to bank for further credit.",
     "category": "rural_livelihoods", "agent_name": "Block OLM Coordinator Nayak"},
    {"query": "Damaged culvert on rural road blocking village access in Bargarh",
     "resolution": "GP Secretary reported to Block office. Emergency repair authorized under Finance Commission untied grants. Temporary bailey bridge placed within 48 hours for vehicle access. Permanent culvert reconstruction sanctioned under BGBO with Rs. 8 lakh estimate. Work completed within 60 days.",
     "category": "infrastructure", "agent_name": "Block JE Patel"},
    {"query": "Nirman Shramik housing application not processed in Sambalpur",
     "resolution": "Block office verified applicant's registration with Building Workers Welfare Board. Found registration had expired. Helped applicant renew registration at Labour office. Fresh housing application submitted through GP and processed within 30 days. Construction assistance sanctioned.",
     "category": "rural_housing", "agent_name": "BDO Suna"},
]

# -------------------------------------------------------------------
# Seed Data: Government Schemes
# -------------------------------------------------------------------
SCHEMES = [
    {"name": "Jal Jeevan Mission (JJM)",
     "description": "Central Government flagship mission to provide Functional Household Tap Connections (FHTC) to every rural household by 2026. Odisha targets 80+ lakh rural households. Includes water quality monitoring, greywater management, and source sustainability. State share 10% for Himalayan/NE states, 50% for others.",
     "eligibility": "All rural households without piped water supply. Priority: SC/ST habitations, water-quality-affected areas, drought-prone villages, Sansad Adarsh Gram Yojana villages.",
     "department": "rural_water_supply",
     "how_to_apply": "Apply at Gram Panchayat office or contact the GP Jalasathi. No application fee. Habitations are covered as per approved Village Action Plan. Contact Block RWSS office or call JJM helpline 1916 for status."},
    {"name": "Basudha (Buxi Jagabandhu Assured Water Supply to Habitations)",
     "description": "State government scheme providing piped drinking water to habitations not covered under JJM. Launched with Rs. 203 crore budget. Uses surface water sources (rivers) for mega piped water supply. Includes 78 water testing laboratories for quality assurance.",
     "eligibility": "Rural habitations in Odisha without adequate safe drinking water, especially in Puri and Ganjam districts initially, expanding statewide. Areas with fluoride, arsenic, or salinity-affected groundwater get priority.",
     "department": "rural_water_supply",
     "how_to_apply": "No individual application needed — coverage is at habitation level. Report water supply issues or request inclusion to GP Sarpanch or Block RWSS. Helpline: 1916."},
    {"name": "MGNREGS (Mahatma Gandhi National Rural Employment Guarantee Scheme)",
     "description": "Provides 100 days of guaranteed wage employment per year to every rural household whose adult members volunteer to do unskilled manual work. Wage rate Rs. 289/day in Odisha (2025-26). Works include: rural roads, water conservation, land development, flood protection, and rural connectivity.",
     "eligibility": "Any adult member of a rural household willing to do unskilled manual work. Must possess a MGNREGS Job Card issued by the Gram Panchayat. No income criteria.",
     "department": "mgnregs",
     "how_to_apply": "Apply for Job Card at Gram Panchayat with family photo and Aadhaar. Card issued within 15 days. Demand work through written application to GP Sarpanch. Work must be provided within 15 days of demand or unemployment allowance is payable."},
    {"name": "PMAY-Gramin (Pradhan Mantri Awas Yojana - Gramin)",
     "description": "Central rural housing scheme providing financial assistance of Rs. 1.20 lakh (plain) / Rs. 1.30 lakh (hilly/difficult areas) for construction of pucca house. Convergence with MGNREGS for 90/95 person-days of unskilled labour. SBM for toilet construction. Three installments released on geo-tagged progress verification.",
     "eligibility": "Households with no pucca house as per SECC 2011 data (Awaas+ list). Priority: Houseless, living in 0/1 room kutcha houses, SC/ST, minorities, freed bonded labourers, widows, persons with disabilities.",
     "department": "rural_housing",
     "how_to_apply": "Check eligibility on Awaas+ portal (awaassoft.nic.in) or at GP office. Selected from SECC permanent wait list approved by Gram Sabha. Required: Aadhaar, bank account. Geo-tagging at foundation, lintel, and completion stages."},
    {"name": "Bikashita Gaon Bikashita Odisha (BGBO)",
     "description": "State flagship rural infrastructure scheme with Rs. 1,000 crore annual allocation. Covers construction and repair of rural roads, bridges, culverts, school buildings, civic amenities, drainage, and sports facilities. Maximum 35% of funds for road projects. 40% allocation reserved for ITDA (tribal) blocks. Up to 5% for innovative community projects.",
     "eligibility": "All Gram Panchayats in Odisha. Projects must have minimum estimate of Rs. 3 lakh. Proposed by Gram Sabha resolution. Incomplete projects from predecessor scheme (AONO) with 20% expenditure can be completed under BGBO.",
     "department": "infrastructure",
     "how_to_apply": "Projects proposed at Gram Sabha meeting. GP passes resolution and submits to Block office. Technical estimate prepared by Block JE. Approval by District-level committee. Citizens can demand specific projects at Gram Sabha."},
    {"name": "Swachh Bharat Mission - Gramin (SBM-G)",
     "description": "National rural sanitation program. Phase II focuses on ODF (Open Defecation Free) sustainability and Solid/Liquid Waste Management (SLWM). Incentive of Rs. 12,000 for Individual Household Latrine (IHHL) construction. Community sanitary complexes, plastic waste management, and faecal sludge management.",
     "eligibility": "Households without toilets (for IHHL incentive). All GPs for community SLWM projects. Priority: BPL households, SC/ST families, women-headed households, persons with disabilities.",
     "department": "sanitation",
     "how_to_apply": "Apply at GP office for IHHL incentive. GP Secretary verifies eligibility. Construct toilet and submit photo evidence. Rs. 12,000 credited to bank account after verification. For SLWM projects, GP passes resolution at Gram Sabha."},
    {"name": "NRLM / Odisha Livelihood Mission (OLM)",
     "description": "National Rural Livelihoods Mission implemented as Odisha Livelihood Mission. Promotes poverty reduction through SHG institution building, financial inclusion, and livelihood enhancement. SHGs receive Revolving Fund (Rs. 15,000) and Community Investment Fund (Rs. 50,000-1,00,000). Bank linkage loans up to Rs. 3+ lakh at subvented interest rates.",
     "eligibility": "Rural BPL women form groups of 10-20 members. Priority: SC/ST households, landless labourers, persons with disabilities, minorities. SHG must maintain regular savings and meetings for 6 months before bank linkage.",
     "department": "rural_livelihoods",
     "how_to_apply": "Contact GP-level Community Resource Person (CRP) or Block OLM Coordinator to join/form an SHG. Required: Aadhaar, bank account for SHG. After 6 months of regular activity, apply for Revolving Fund through Block office."},
    {"name": "15th Finance Commission Grants to PRIs",
     "description": "Central grants to Gram Panchayats, Panchayat Samitis, and Zilla Parishads for basic civic services. 50% tied grants mandatory for drinking water/sanitation (including rainwater harvesting, water recycling). 50% untied grants for any GP-prioritized development. Total allocation for Odisha: Rs. 1,002+ crore for water and sanitation alone.",
     "eligibility": "All three-tier PRIs (Gram Panchayat, Panchayat Samiti, Zilla Parishad) in Odisha. Utilization must be as per approved action plan. Audit and utilization certificates mandatory for next installment.",
     "department": "infrastructure",
     "how_to_apply": "No citizen application needed — grants are released to PRIs automatically. Citizens can influence utilization through Gram Sabha resolutions. Track fund flow at egramswaraj.gov.in. Report misuse to BDO or District Panchayat Officer."},
    {"name": "Nirman Shramik Pucca Ghar Yojana",
     "description": "State housing scheme specifically for registered construction workers. Provides financial assistance for pucca house construction. Implemented through Panchayati Raj Department in convergence with Odisha Building Workers Welfare Board.",
     "eligibility": "Must be registered with Odisha Building and Construction Workers Welfare Board. Must not own a pucca house. Registration must be active (renewed within 3 years).",
     "department": "rural_housing",
     "how_to_apply": "Apply at GP or Block office with construction worker registration certificate, Aadhaar, land ownership/patta document, and bank passbook. GP Secretary verifies and forwards to Block. Sanctioned amount released in installments."},
    {"name": "Mission Shakti (SHG Convergence through PR&DW)",
     "description": "Women's SHG empowerment program with convergence through PR&DW Department. SHGs formed under Mission Shakti are linked to MGNREGS works, JJM operations (Jalasathi), GP-level service delivery, and BGBO project monitoring. Interest-free loans up to Rs. 3 lakh for livelihood activities.",
     "eligibility": "Women Self Help Groups registered under Mission Shakti / NRLM. Individual members age 18+ years. SHGs must have minimum 6 months of regular savings and meeting records.",
     "department": "rural_livelihoods",
     "how_to_apply": "Contact Block Mission Shakti / OLM office. SHG registration through CRP at GP level. Required: SHG registration, member Aadhaar cards, joint bank account. Loan applications through bank linkage after grading."},
]

# -------------------------------------------------------------------
# Seed Data: Test Grievances
# -------------------------------------------------------------------
GRIEVANCES = [
    # --- RESOLVED: AI self-resolved (citizen confirmed helpful) ---
    {"title": "How to check PMAY-G application status online",
     "description": "I want to know how to check my PMAY-G housing application status online. I applied 2 months ago at the GP office. I do not know the website or process. Can someone guide me?",
     "district": "Puri", "department": "rural_housing", "priority": "low",
     "sentiment": "neutral", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "ai",
     "ai_resolution": "You can check your PMAY-G application status online through the **Awaas+ portal**:\n\n1. Visit [awaassoft.nic.in](https://awaassoft.nic.in)\n2. Click on **'Stakeholders'** → **'IAY/PMAY-G Beneficiary'**\n3. Enter your **Registration Number** (given at GP office) or search by name\n4. Your application status, installment details, and geo-tag progress will be displayed\n\nAlternatively, you can call the **PMAY-G helpline: 1800-11-6446** (toll-free) for status updates.\n\nIf you don't have your registration number, visit your Gram Panchayat office — the GP Secretary can look it up using your Aadhaar number.",
     "confidence_score": 0.92, "resolution_feedback": 5,
     "citizen_name": "Ganesh Behera", "citizen_email": "ganesh.behera@email.com"},

    # --- RESOLVED: AI self-resolved (citizen confirmed helpful) ---
    {"title": "What documents are needed for MGNREGS job card application",
     "description": "I want to apply for MGNREGS job card. What documents do I need to bring to the GP office? My neighbor said they need many papers.",
     "district": "Khordha", "department": "mgnregs", "priority": "low",
     "sentiment": "neutral", "status": "resolved", "resolution_tier": "self_resolvable",
     "resolution_type": "ai",
     "ai_resolution": "To apply for a **MGNREGS Job Card**, you need the following documents:\n\n### Required Documents\n1. **Aadhaar Card** of the head of household\n2. **Family photograph** (passport size, with all adult members)\n3. **Address proof** (Aadhaar/Voter ID/Ration card)\n\n### Process\n- Visit your **Gram Panchayat office** and submit a written application\n- The GP Secretary will process it within **15 days**\n- Job card is issued **free of cost** — no fee is required\n- Once issued, you can demand work by giving written application to the Sarpanch\n\n### Important\n- **No income criteria** — every rural household is eligible\n- If the card is not issued within 15 days, complain to the Block MGNREGS Programme Officer\n- Current wage rate in Odisha: **Rs. 289/day** (2025-26)",
     "confidence_score": 0.95, "resolution_feedback": 4,
     "citizen_name": "Prasanna Sahoo", "citizen_email": "prasanna.sahoo@email.com"},

    # --- RESOLVED: Officer manual resolution ---
    {"title": "MGNREGS wages not paid for 60 days in Kalahandi",
     "description": "I and 30 other workers from our village completed road construction work under MGNREGS 60 days ago. Muster rolls were signed. But wages have not been credited to our bank accounts. The GP Sarpanch says he does not know the reason.",
     "district": "Kalahandi", "department": "mgnregs", "priority": "high",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual",
     "manual_resolution": "**Investigation completed by Block MGNREGS Programme Officer.**\n\nThe Fund Transfer Order (FTO) was stuck due to incorrect bank account details for 23 out of 31 workers. The account numbers had data entry errors at the GP level.\n\n**Actions taken:**\n1. Aadhaar-based re-verification of all 31 worker bank accounts completed\n2. Corrected FTO resubmitted to State MGNREGS cell\n3. Wages for 60 days (**Rs. 17,340 per worker**) transferred on 28-Jan-2026\n4. Delay compensation @ 0.05% per day also credited\n5. GP Secretary issued warning for negligence in data entry\n\nAll 31 workers have confirmed receipt of wages. Case closed.",
     "assigned_officer": "Smt. Priya Pattnaik, BDO", "resolution_feedback": 5,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "FTO investigation started. Found bank account mismatch for 23 workers.", "note_type": "internal"},
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "All accounts corrected via Aadhaar re-verification. FTO resubmitted.", "note_type": "citizen_facing"},
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Wages credited to all 31 accounts. Delay compensation included.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Laxman Naik", "citizen_email": "laxman.naik@email.com"},

    # --- RESOLVED: Officer manual resolution ---
    {"title": "Bore well non-functional for 4 months in tribal village",
     "description": "The only bore well in our tribal hamlet of 60 families in Mayurbhanj has not been working for 4 months. The hand pump handle is broken and motor is burned out. Women and children walk 3 km daily to fetch water from a stream.",
     "district": "Mayurbhanj", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual",
     "manual_resolution": "**RWSS Junior Engineer inspection report:**\n\nSite inspection conducted on 05-Jan-2026. Found:\n- Submersible pump motor completely burned out (beyond repair)\n- Hand pump handle fractured at pivot joint\n\n**Repair actions:**\n1. New 1.5 HP submersible pump motor installed under Basudha maintenance budget — **completed 08-Jan-2026**\n2. Hand pump handle replaced with heavy-duty stainless steel model — **completed 08-Jan-2026**\n3. GP Jalasathi trained on basic pump maintenance and monthly inspection checklist\n4. Quarterly preventive maintenance schedule established with RWSS Block office\n\nWater supply restored to all 60 households. Follow-up visit scheduled for March 2026.",
     "assigned_officer": "Smt. Priya Pattnaik, BDO", "resolution_feedback": 5,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Urgent: Assigned RWSS JE for immediate site inspection given 60 families affected.", "note_type": "internal"},
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Repair completed. Water supply restored. Jalasathi training done.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Gurubari Hansda", "citizen_email": "gurubari.hansda@email.com"},

    # --- RESOLVED: Hybrid (AI draft approved by officer) ---
    {"title": "SBM toilet subsidy not received after construction completed",
     "description": "I constructed a toilet under SBM-G scheme 5 months ago. The GP Secretary took photos and said the incentive of Rs. 12,000 will be credited. But nothing has been received. When I ask, they say it is under processing.",
     "district": "Balasore", "department": "sanitation", "priority": "medium",
     "sentiment": "negative", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "hybrid",
     "ai_resolution": "Based on similar resolved cases, the most common cause of delayed SBM-G incentive payments is that the beneficiary was not registered in the IMIS (Information Management and Monitoring System) portal. The GP Secretary needs to complete the online entry with photo evidence before payment can be processed.",
     "manual_resolution": "**Block SBM Coordinator verified the complaint.**\n\nInvestigation confirmed the AI assessment — beneficiary construction was verified physically but the IMIS portal entry was incomplete. GP Secretary had uploaded photos but did not complete the verification form.\n\n**Resolution:**\n1. IMIS entry completed by Block SBM Coordinator on 15-Jan-2026\n2. Incentive of **Rs. 12,000** credited to beneficiary bank account (A/C ending 4521) on 22-Jan-2026\n3. GP Secretary counseled on proper IMIS documentation procedure\n\nPayment confirmed by beneficiary.",
     "assigned_officer": "Smt. Priya Pattnaik, BDO", "confidence_score": 0.78,
     "resolution_feedback": 4,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "AI draft was accurate — IMIS entry was indeed incomplete. Approved and supplemented with specific resolution details.", "note_type": "internal"},
     ],
     "citizen_name": "Basanti Jena", "citizen_email": "basanti.jena@email.com"},

    # --- RESOLVED: Officer manual ---
    {"title": "Water from JJM tap is yellow and smells bad",
     "description": "We received JJM tap connections 2 months ago in our village in Dhenkanal. But the water is yellowish with a metallic smell. Children have been getting stomach problems. We suspect high iron content. Please test the water quality.",
     "district": "Dhenkanal", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "resolved", "resolution_tier": "officer_action",
     "resolution_type": "manual",
     "manual_resolution": "**Water quality test results and remediation:**\n\nRWSS collected samples from 5 points in the pipeline network on 10-Jan-2026.\n\n**Lab Results:**\n| Parameter | Found | Safe Limit |\n|-----------|-------|------------|\n| Iron | 3.2 mg/L | 0.3 mg/L |\n| Turbidity | 8 NTU | 5 NTU |\n| pH | 6.8 | 6.5-8.5 |\n| Bacteriological | Safe | — |\n\n**High iron content confirmed (10x above safe limit).**\n\n**Remediation:**\n1. Iron Removal Plant (IRP) ordered under JJM water quality component — **installed 28-Jan-2026**\n2. Pipeline flushing done to clear accumulated deposits\n3. Follow-up water testing on 02-Feb-2026 confirmed iron levels at **0.2 mg/L** (within safe limits)\n4. Quarterly water quality monitoring schedule established\n\nWater is now safe for consumption. Residents advised to report any recurrence immediately.",
     "assigned_officer": "Smt. Priya Pattnaik, BDO", "resolution_feedback": 5,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "URGENT: Water samples collected from 5 points. Sent to district RWSS lab for testing.", "note_type": "internal"},
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Lab results confirm high iron. IRP ordered under JJM water quality budget.", "note_type": "citizen_facing"},
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "IRP installed and operational. Follow-up test confirms safe water quality.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Surekha Pradhan", "citizen_email": "surekha.pradhan@email.com"},

    # --- IN PROGRESS: Assigned, being worked on ---
    {"title": "No JJM tap water connection despite being in village action plan",
     "description": "Our habitation of 85 families in Koraput block was included in the JJM Village Action Plan last year. Pipeline work started but stopped 2 months ago. No tap connections provided yet. We still depend on a contaminated open well.",
     "district": "Koraput", "department": "rural_water_supply", "priority": "urgent",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "assigned_officer": "Smt. Priya Pattnaik, BDO",
     "ai_resolution": "Based on similar cases in Koraput district, pipeline work stoppages are commonly caused by: (1) rocky terrain requiring percussion drilling, (2) contractor payment disputes, or (3) Right-of-Way clearance issues. Recommend contacting the Block RWSS Executive Engineer for specific status on the pipeline extension work.",
     "confidence_score": 0.65,
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Contacted RWSS EE Koraput. Contractor reports rock formations requiring special equipment. Percussion drilling team being mobilized.", "note_type": "internal"},
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Your complaint has been escalated to the RWSS Executive Engineer, Koraput. Special drilling equipment is being arranged to complete the pipeline through rocky terrain. Expected completion: 4-6 weeks.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Dambaru Majhi", "citizen_email": "dambaru.majhi@email.com"},

    # --- IN PROGRESS: Assigned, investigation underway ---
    {"title": "BGBO road project abandoned halfway in Rayagada block",
     "description": "A 3 km rural road sanctioned under BGBO from our village to the main road has been abandoned after only 1 km of construction. The contractor left the site 4 months ago. During monsoon the unfinished road becomes impassable and dangerous.",
     "district": "Rayagada", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "assigned_officer": "Smt. Priya Pattnaik, BDO",
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "DPO field inspection completed. Contractor abandoned at 60% citing material cost escalation. Contract termination and penalty proceedings initiated.", "note_type": "internal"},
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "The contractor's performance has been found unsatisfactory. Contract termination process has been started and re-tendering will begin shortly. A new contractor will be engaged to complete the remaining 2 km.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Kamala Sabar", "citizen_email": "kamala.sabar@email.com"},

    # --- IN PROGRESS: Assigned ---
    {"title": "Damaged culvert blocking school access in Kandhamal village",
     "description": "A culvert on the village road in Kandhamal collapsed during heavy rains last month. Now children cannot cross to reach the school on the other side. Vehicles also cannot pass. We need emergency repair under BGBO or Finance Commission grants.",
     "district": "Kandhamal", "department": "infrastructure", "priority": "urgent",
     "sentiment": "frustrated", "status": "in_progress", "resolution_tier": "officer_action",
     "assigned_officer": "Smt. Priya Pattnaik, BDO",
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Emergency: Temporary bailey bridge placed within 48 hours for vehicle and pedestrian access. Permanent culvert reconstruction sanctioned under BGBO with Rs. 8 lakh estimate.", "note_type": "citizen_facing"},
     ],
     "citizen_name": "Gobinda Kanhar", "citizen_email": "gobinda.kanhar@email.com"},

    # --- ESCALATED: Serious allegations ---
    {"title": "Finance Commission grant funds allegedly misused by Sarpanch",
     "description": "In our GP in Kendrapara, the Sarpanch has used Finance Commission untied grant money to construct a boundary wall around his own property. The community demanded a road and drain but was ignored. Villagers have proof including photos and payment vouchers.",
     "district": "Kendrapara", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Complaint forwarded to DPO for audit. Photographs and voucher copies received from complainant.", "note_type": "internal"},
     ],
     "citizen_name": None, "citizen_email": None, "is_anonymous": True},

    # --- ESCALATED: Corruption allegation ---
    {"title": "Sarpanch demanding bribe for PMAY-G beneficiary selection",
     "description": "The Sarpanch of our GP in Malkangiri is demanding Rs. 10,000 from each family to include them in the PMAY-G beneficiary list. Many deserving families who cannot pay are being left out. This is corruption and must be investigated.",
     "district": "Malkangiri", "department": "rural_housing", "priority": "urgent",
     "sentiment": "frustrated", "status": "escalated", "resolution_tier": "escalation",
     "notes": [
         {"officer": "Smt. Priya Pattnaik, BDO", "content": "Serious corruption allegation. Referred to District Collector and Vigilance cell for investigation. PMAY-G beneficiary list frozen pending inquiry.", "note_type": "internal"},
     ],
     "citizen_name": None, "citizen_email": None, "is_anonymous": True},

    # --- PENDING: Officer action needed ---
    {"title": "PMAY-G second installment not released despite completing lintel",
     "description": "I completed construction up to lintel level for my PMAY-G house 3 months ago. Geo-tagged photos were taken by the Block TA. But second installment of Rs. 40,000 has not been released. I have exhausted my savings and cannot continue construction.",
     "district": "Ganjam", "department": "rural_housing", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "officer_action",
     "citizen_name": "Somanath Behera", "citizen_email": "somanath.behera@email.com"},

    # --- PENDING: With AI draft awaiting officer review ---
    {"title": "NRLM SHG bank linkage loan pending for 8 months",
     "description": "Our SHG 'Maa Tarini' in Nabarangpur has been regularly saving and meeting for 2 years. We applied for bank linkage loan of Rs. 2 lakh 8 months ago. The bank keeps asking for more documents. The Block OLM office says they have forwarded it.",
     "district": "Nabarangpur", "department": "rural_livelihoods", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "ai_resolution": "Based on similar resolved cases in Nabarangpur district, the most common reason for delayed SHG bank linkage is incomplete grading documentation. The Block OLM Coordinator should:\n\n1. Verify that the SHG credit rating/grading exercise has been completed\n2. Check if monthly meeting minutes are properly documented for all 24 months\n3. Coordinate directly with the bank branch manager to identify specific pending documents\n4. If the bank is non-cooperative, escalate to the Lead District Manager (LDM) under NRLM banking correspondent guidelines\n\nTypical resolution timeline after intervention: 15-20 days.",
     "confidence_score": 0.72,
     "citizen_name": "Sabita Nayak", "citizen_email": "sabita.nayak@email.com"},

    # --- PENDING: With AI draft awaiting officer review ---
    {"title": "BGBO community hall construction using substandard material",
     "description": "A community hall is being built under BGBO scheme in our village in Sundargarh. The contractor is using poor quality bricks and less cement. Cracks have already appeared in the walls. We reported to the Block JE but no action taken.",
     "district": "Sundargarh", "department": "infrastructure", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "escalation",
     "ai_resolution": "This is a serious quality control issue that requires immediate field investigation. Recommended actions:\n\n1. **Block JE must conduct a quality assessment** including concrete cube test and material verification\n2. If substandard material is confirmed, the contractor should be directed to **demolish and rebuild** the affected portion at their own cost\n3. A **penalty of 10% should be deducted** from the contractor's payment as per BGBO guidelines\n4. **Third-party quality monitoring** should be engaged for the remaining work\n\nThis matter should be escalated to the District Panchayat Officer if the Block JE does not act within 7 days.",
     "confidence_score": 0.68,
     "citizen_name": "Mangal Kisan", "citizen_email": "mangal.kisan@email.com"},

    # --- PENDING: Self-resolvable (awaiting citizen confirmation) ---
    {"title": "How to register complaint about non-functional street light",
     "description": "The solar street lights installed by GP in our village stopped working 2 weeks ago. Where do I complain? Whom should I contact?",
     "district": "Cuttack", "department": "panchayati_raj", "priority": "low",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "self_resolvable",
     "resolution_type": "ai",
     "ai_resolution": "For non-functional solar street lights installed by the Gram Panchayat, here's how to get them fixed:\n\n### Step 1: Contact GP Office\n- Report to your **GP Secretary** or **Sarpanch** in writing\n- Solar street lights installed by GP are maintained under **Finance Commission grants**\n\n### Step 2: Warranty Check\n- Most solar street lights have a **5-year warranty** from the manufacturer\n- Ask the GP Secretary to check if the lights are still under warranty\n- If under warranty, the manufacturer must repair/replace at no cost\n\n### Step 3: Escalation\n- If GP doesn't act within 7 days, contact the **Block Development Officer (BDO)**\n- BDO contact details available at [panchayat.odisha.gov.in](https://panchayat.odisha.gov.in)\n\n### Step 4: Alternative Reporting\n- Call the **PR&DW Department helpline** for your district\n- You can also report on the **e-Gram Swaraj portal** (egramswaraj.gov.in)",
     "confidence_score": 0.88,
     "citizen_name": "Sudhir Mohanty", "citizen_email": "sudhir.mohanty@email.com"},

    # --- PENDING: Fresh, no action yet ---
    {"title": "Gram Panchayat has not held Gram Sabha for over 1 year",
     "description": "Our Gram Panchayat in Boudh district has not conducted any Gram Sabha since January 2025. Neither the mandatory Republic Day nor Gandhi Jayanti sessions were held. The Sarpanch is not responsive to requests. Beneficiary lists are being decided without public consultation.",
     "district": "Boudh", "department": "panchayati_raj", "priority": "high",
     "sentiment": "frustrated", "status": "pending", "resolution_tier": "escalation",
     "citizen_name": "Brundaban Sahu", "citizen_email": "brundaban.sahu@email.com"},

    # --- PENDING: Fresh ---
    {"title": "JJM pipeline laid but no water flowing for 2 months",
     "description": "JJM pipeline was laid to our habitation in Angul 2 months ago and taps were installed in all houses. But no water has ever flowed through the taps. The overhead tank was built but seems not connected. We still use the old tube well.",
     "district": "Angul", "department": "rural_water_supply", "priority": "high",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "citizen_name": "Bijay Kumar Sahu", "citizen_email": "bijay.sahu@email.com"},

    # --- PENDING: Fresh ---
    {"title": "MGNREGS job card not issued despite applying 3 months ago",
     "description": "I applied for a MGNREGS Job Card at the GP office in Nuapada 3 months ago with all required documents including family photo and Aadhaar. The GP Secretary says it is under processing. Without the card, I cannot demand work or earn wages.",
     "district": "Nuapada", "department": "mgnregs", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "citizen_name": "Ratan Majhi", "citizen_email": "ratan.majhi@email.com"},

    # --- PENDING: Fresh ---
    {"title": "Request for new bore well under Basudha in Sundargarh village",
     "description": "Our village of 150 families in Sundargarh has only one hand pump which dries up in summer. We face severe water crisis from March to June every year. We request installation of a new bore well under Basudha scheme. The GP has passed a resolution.",
     "district": "Sundargarh", "department": "rural_water_supply", "priority": "medium",
     "sentiment": "neutral", "status": "pending", "resolution_tier": "officer_action",
     "citizen_name": "Birsa Munda", "citizen_email": "birsa.munda@email.com"},

    # --- PENDING: Fresh ---
    {"title": "SBM community waste management not functioning in Jajpur GP",
     "description": "A Solid and Liquid Waste Management (SLWM) unit was installed in our GP under SBM-G last year. It has not been operational since installation. Waste is piling up near the unit. No staff assigned for operation. The village is losing its ODF Plus status.",
     "district": "Jajpur", "department": "sanitation", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "citizen_name": "Lalita Sahoo", "citizen_email": "lalita.sahoo@email.com"},

    # --- PENDING: Fresh ---
    {"title": "MGNREGS worksite has no shade or drinking water facility",
     "description": "At the MGNREGS worksite near our village in Bargarh, there is no shade shelter, no drinking water, and no first-aid kit. Women workers are suffering in the heat. The mate says there is no budget for facilities. This violates MGNREGS guidelines.",
     "district": "Bargarh", "department": "mgnregs", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "citizen_name": "Pramila Sahu", "citizen_email": "pramila.sahu@email.com"},

    # --- PENDING: Fresh ---
    {"title": "Mission Shakti SHG not receiving revolving fund in Gajapati",
     "description": "Our SHG 'Maa Lakshmi' in Gajapati was formed under Mission Shakti 1 year ago. We have been saving Rs. 100/month regularly and conducting weekly meetings. But the Revolving Fund of Rs. 15,000 has not been received. The Block OLM Coordinator says funds are exhausted.",
     "district": "Gajapati", "department": "rural_livelihoods", "priority": "medium",
     "sentiment": "negative", "status": "pending", "resolution_tier": "officer_action",
     "citizen_name": "Kuni Sabar", "citizen_email": "kuni.sabar@email.com"},
]

# -------------------------------------------------------------------
# Seed Data: Users
# -------------------------------------------------------------------
USERS = [
    {"username": "citizen1", "password": "citizen123", "full_name": "Rajesh Kumar Swain",
     "email": "rajesh.swain@email.com", "phone": "9876543210", "role": "citizen", "department": None},
    {"username": "officer1", "password": "officer123", "full_name": "Smt. Priya Pattnaik, BDO",
     "email": "priya.bdo@panchayat.odisha.gov.in", "phone": "9988776655", "role": "officer", "department": "panchayati_raj"},
    {"username": "admin", "password": "admin123", "full_name": "System Administrator",
     "email": "admin@panchayat.odisha.gov.in", "phone": None, "role": "admin", "department": None},
]

# -------------------------------------------------------------------
# Import Logic
# -------------------------------------------------------------------
async def get_embedding(text: str):
    response = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

async def main():
    print("=" * 60)
    print("PR&DW Grievance Portal - Data Importer")
    print("=" * 60)

    # Connect MongoDB
    print("\n[1/6] Connecting to MongoDB...")
    mongo_client = MongoClient(MONGODB_URL)
    db = mongo_client.grievance_system
    print(f"  Connected to {MONGODB_URL}")

    # Connect Qdrant
    print("\n[2/6] Connecting to Qdrant...")
    if QDRANT_API_KEY:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    else:
        qdrant = QdrantClient(url=QDRANT_URL)
    print(f"  Connected to {QDRANT_URL}")

    # Reset collections
    print("\n  Resetting collections...")
    for coll_name in ["documentation", "service_memory", "schemes"]:
        try:
            qdrant.delete_collection(coll_name)
        except:
            pass
        qdrant.create_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print(f"    Created Qdrant collection: {coll_name}")

    db.grievances.drop()
    db.users.drop()
    db.counters.drop()
    print("    Reset MongoDB collections: grievances, users, counters")

    # Import users
    print("\n[3/6] Importing seed users...")
    for u in USERS:
        user_doc = {
            "_id": str(uuid.uuid4()),
            "username": u["username"],
            "hashed_password": pwd_context.hash(u["password"]),
            "full_name": u["full_name"],
            "email": u["email"],
            "phone": u["phone"],
            "role": u["role"],
            "department": u["department"],
            "created_at": datetime.now(timezone.utc),
        }
        db.users.insert_one(user_doc)
        print(f"  Created user: {u['username']} ({u['role']})")
    db.users.create_index([("username", 1)], unique=True)

    # Import documentation
    print("\n[4/6] Importing documentation...")
    for i, doc in enumerate(DOCUMENTATION):
        print(f"  [{i+1}/{len(DOCUMENTATION)}] {doc['title'][:50]}...")
        embedding = await get_embedding(doc["content"])
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"title": doc["title"], "content": doc["content"],
                     "category": doc["category"], "created_at": datetime.now(timezone.utc).isoformat()}
        )
        qdrant.upsert(collection_name="documentation", points=[point], wait=True)
    print(f"  Imported {len(DOCUMENTATION)} documentation entries")

    # Import service memory
    print("\n[5/6] Importing service memory...")
    for i, mem in enumerate(SERVICE_MEMORY):
        print(f"  [{i+1}/{len(SERVICE_MEMORY)}] {mem['query'][:50]}...")
        embedding = await get_embedding(mem["query"])
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"query": mem["query"], "resolution": mem["resolution"],
                     "category": mem["category"], "agent_name": mem["agent_name"],
                     "created_at": datetime.now(timezone.utc).isoformat()}
        )
        qdrant.upsert(collection_name="service_memory", points=[point], wait=True)
    print(f"  Imported {len(SERVICE_MEMORY)} service memory entries")

    # Import schemes
    print("\n  Importing government schemes...")
    for i, s in enumerate(SCHEMES):
        print(f"  [{i+1}/{len(SCHEMES)}] {s['name'][:50]}...")
        text = f"{s['name']} {s['description']} {s['eligibility']} {s['how_to_apply']}"
        embedding = await get_embedding(text)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"name": s["name"], "description": s["description"],
                     "eligibility": s["eligibility"], "department": s["department"],
                     "how_to_apply": s["how_to_apply"],
                     "created_at": datetime.now(timezone.utc).isoformat()}
        )
        qdrant.upsert(collection_name="schemes", points=[point], wait=True)
    print(f"  Imported {len(SCHEMES)} government schemes")

    # Import grievances
    print("\n[6/6] Importing test grievances...")
    # Look up citizen1 so we can link grievances to their account
    citizen1 = db.users.find_one({"username": "citizen1"})
    citizen1_id = str(citizen1["_id"]) if citizen1 else None

    officer_cats = {
        "panchayati_raj": "block_dev_officer",
        "rural_water_supply": "executive_engineer_rwss",
        "mgnregs": "mgnregs_programme_officer",
        "rural_housing": "block_dev_officer",
        "rural_livelihoods": "drda_project_director",
        "sanitation": "block_dev_officer",
        "infrastructure": "block_dev_officer",
        "general": "general_officer",
    }
    now = datetime.now(timezone.utc)
    for i, g in enumerate(GRIEVANCES):
        seq = db.counters.find_one_and_update(
            {"_id": "grievance"}, {"$inc": {"seq": 1}},
            upsert=True, return_document=True
        )
        tracking = f"GRV-{now.year}-{seq['seq']:06d}"
        is_anon = g.get("is_anonymous", False)
        priority_hours = {"low": 360, "medium": 168, "high": 72, "urgent": 24}
        created = now - timedelta(days=25 - i)
        status = g["status"]

        if status == "resolved":
            updated = created + timedelta(days=3, hours=i * 2)
            sla = created + timedelta(hours=priority_hours.get(g["priority"], 168))
        elif status == "in_progress":
            updated = now - timedelta(days=2, hours=i)
            sla = now + timedelta(hours=priority_hours.get(g["priority"], 168))
        elif status == "escalated":
            updated = now - timedelta(days=1)
            sla = now + timedelta(hours=12)
        else:
            updated = created + timedelta(hours=i)
            sla = now + timedelta(hours=priority_hours.get(g["priority"], 168))

        seed_notes = []
        for n in g.get("notes", []):
            seed_notes.append({
                "officer": n["officer"],
                "content": n["content"],
                "note_type": n["note_type"],
                "created_at": created + timedelta(days=1, hours=len(seed_notes) * 6),
            })

        doc = {
            "_id": str(uuid.uuid4()),
            "tracking_number": tracking,
            "title": g["title"],
            "description": g["description"],
            "citizen_name": None if is_anon else g.get("citizen_name"),
            "citizen_email": None if is_anon else g.get("citizen_email"),
            "citizen_phone": None,
            "is_anonymous": is_anon,
            "language": "english",
            "district": g.get("district"),
            "department": g["department"],
            "priority": g["priority"],
            "officer_category": officer_cats.get(g["department"], "general_officer"),
            "status": status,
            "sentiment": g["sentiment"],
            "created_at": created,
            "updated_at": updated,
            "sla_deadline": sla,
            "ai_resolution": g.get("ai_resolution"),
            "manual_resolution": g.get("manual_resolution"),
            "resolution_tier": g.get("resolution_tier", "officer_action"),
            "resolution_type": g.get("resolution_type"),
            "confidence_score": g.get("confidence_score", 0.0),
            "assigned_officer": g.get("assigned_officer"),
            "resolution_feedback": g.get("resolution_feedback"),
            "notes": seed_notes,
            "location": g.get("location"),
            "citizen_user_id": citizen1_id,
        }
        db.grievances.insert_one(doc)
        tag = {"resolved": "✅", "in_progress": "🔄", "escalated": "⚠️", "pending": "⏳"}.get(status, "")
        print(f"  [{i+1}/{len(GRIEVANCES)}] {tag} {tracking}: {g['title'][:50]}...")

    # Create indexes
    db.grievances.create_index("created_at")
    db.grievances.create_index("status")
    db.grievances.create_index("department")
    db.grievances.create_index("priority")
    db.grievances.create_index("tracking_number")

    print(f"\n  Imported {len(GRIEVANCES)} test grievances")

    # Summary
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"  Documentation:  {len(DOCUMENTATION)} entries")
    print(f"  Service Memory: {len(SERVICE_MEMORY)} entries")
    print(f"  Schemes:        {len(SCHEMES)} entries")
    print(f"  Grievances:     {len(GRIEVANCES)} entries")
    print(f"  Users:          {len(USERS)} entries")
    print(f"\nTest credentials:")
    print(f"  Citizen: citizen1 / citizen123")
    print(f"  Officer: officer1 / officer123")
    print(f"  Admin:   admin / admin123")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
