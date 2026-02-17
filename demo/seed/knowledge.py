# Seed data: Knowledge Base (Documentation + Service Memory)

from qdrant_client.models import PointStruct

from .config import new_id, now_utc, get_embedding

# ---------------------------------------------------------------------------
# Documentation (15 entries — carried forward from original importer)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Service Memory (16 entries — carried forward from original importer)
# ---------------------------------------------------------------------------
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
    {"query": "MGNREGS worksite lacks mandated facilities — no shade, water, or first-aid in Bargarh",
     "resolution": "Block MGNREGS PO inspected the worksite. Confirmed absence of mandatory facilities. Contractor issued notice. Temporary shade erected within 24 hours. Drinking water tanker arranged. First-aid kit procured from Block health centre. GP-level monitoring committee activated per MGNREGS operational guidelines.",
     "category": "mgnregs", "agent_name": "BPO Panda"},
]

# ---------------------------------------------------------------------------
# Import functions
# ---------------------------------------------------------------------------
async def import_documentation(qdrant) -> int:
    print("\n  Importing documentation...")
    for i, doc in enumerate(DOCUMENTATION):
        print(f"    [{i+1}/{len(DOCUMENTATION)}] {doc['title'][:55]}...")
        embedding = await get_embedding(doc["content"])
        point = PointStruct(
            id=new_id(), vector=embedding,
            payload={"title": doc["title"], "content": doc["content"],
                     "category": doc["category"],
                     "created_at": now_utc().isoformat()})
        qdrant.upsert(collection_name="documentation", points=[point], wait=True)
    print(f"  => {len(DOCUMENTATION)} documentation entries")
    return len(DOCUMENTATION)


async def import_service_memory(qdrant) -> int:
    print("\n  Importing service memory...")
    for i, mem in enumerate(SERVICE_MEMORY):
        print(f"    [{i+1}/{len(SERVICE_MEMORY)}] {mem['query'][:55]}...")
        embedding = await get_embedding(mem["query"])
        point = PointStruct(
            id=new_id(), vector=embedding,
            payload={"query": mem["query"], "resolution": mem["resolution"],
                     "category": mem["category"], "agent_name": mem["agent_name"],
                     "created_at": now_utc().isoformat()})
        qdrant.upsert(collection_name="service_memory", points=[point], wait=True)
    print(f"  => {len(SERVICE_MEMORY)} service memory entries")
    return len(SERVICE_MEMORY)
