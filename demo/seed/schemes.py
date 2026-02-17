# Seed data: Government Schemes (12 schemes, each with eligibility_questions)

from qdrant_client.models import PointStruct

from .config import new_id, now_utc, get_embedding

# ---------------------------------------------------------------------------
# Schemes — original 10 + 2 new, all with eligibility_questions
# ---------------------------------------------------------------------------
SCHEMES = [
    # 1
    {"name": "Jal Jeevan Mission (JJM)",
     "description": "Central Government flagship mission to provide Functional Household Tap Connections (FHTC) to every rural household by 2026. Odisha targets 80+ lakh rural households. Includes water quality monitoring, greywater management, and source sustainability. State share 10% for Himalayan/NE states, 50% for others.",
     "eligibility": "All rural households without piped water supply. Priority: SC/ST habitations, water-quality-affected areas, drought-prone villages, Sansad Adarsh Gram Yojana villages.",
     "department": "rural_water_supply",
     "how_to_apply": "Apply at Gram Panchayat office or contact the GP Jalasathi. No application fee. Habitations are covered as per approved Village Action Plan. Contact Block RWSS office or call JJM helpline 1916 for status.",
     "eligibility_questions": [
         {"question": "Is your household located in a rural area?", "eligible_answer": "yes"},
         {"question": "Does your household currently have a piped water connection?", "eligible_answer": "no"},
         {"question": "Is your habitation included in the Village Action Plan?", "eligible_answer": "yes"},
     ]},

    # 2
    {"name": "Basudha (Buxi Jagabandhu Assured Water Supply to Habitations)",
     "description": "State government scheme providing piped drinking water to habitations not covered under JJM. Launched with Rs. 203 crore budget. Uses surface water sources (rivers) for mega piped water supply. Includes 78 water testing laboratories for quality assurance.",
     "eligibility": "Rural habitations in Odisha without adequate safe drinking water, especially in Puri and Ganjam districts initially, expanding statewide. Areas with fluoride, arsenic, or salinity-affected groundwater get priority.",
     "department": "rural_water_supply",
     "how_to_apply": "No individual application needed — coverage is at habitation level. Report water supply issues or request inclusion to GP Sarpanch or Block RWSS. Helpline: 1916.",
     "eligibility_questions": [
         {"question": "Is your habitation in a rural area of Odisha?", "eligible_answer": "yes"},
         {"question": "Does your area have adequate safe drinking water currently?", "eligible_answer": "no"},
         {"question": "Is your area affected by fluoride, arsenic, or salinity in groundwater?", "eligible_answer": "yes"},
     ]},

    # 3
    {"name": "MGNREGS (Mahatma Gandhi National Rural Employment Guarantee Scheme)",
     "description": "Provides 100 days of guaranteed wage employment per year to every rural household whose adult members volunteer to do unskilled manual work. Wage rate Rs. 289/day in Odisha (2025-26). Works include: rural roads, water conservation, land development, flood protection, and rural connectivity.",
     "eligibility": "Any adult member of a rural household willing to do unskilled manual work. Must possess a MGNREGS Job Card issued by the Gram Panchayat. No income criteria.",
     "department": "mgnregs",
     "how_to_apply": "Apply for Job Card at Gram Panchayat with family photo and Aadhaar. Card issued within 15 days. Demand work through written application to GP Sarpanch. Work must be provided within 15 days of demand or unemployment allowance is payable.",
     "eligibility_questions": [
         {"question": "Are you a resident of a rural area?", "eligible_answer": "yes"},
         {"question": "Are you 18 years of age or older?", "eligible_answer": "yes"},
         {"question": "Are you willing to do unskilled manual work?", "eligible_answer": "yes"},
         {"question": "Do you have a MGNREGS Job Card (or are you applying for one)?", "eligible_answer": "yes"},
     ]},

    # 4
    {"name": "PMAY-Gramin (Pradhan Mantri Awas Yojana - Gramin)",
     "description": "Central rural housing scheme providing financial assistance of Rs. 1.20 lakh (plain) / Rs. 1.30 lakh (hilly/difficult areas) for construction of pucca house. Convergence with MGNREGS for 90/95 person-days of unskilled labour. SBM for toilet construction. Three installments released on geo-tagged progress verification.",
     "eligibility": "Households with no pucca house as per SECC 2011 data (Awaas+ list). Priority: Houseless, living in 0/1 room kutcha houses, SC/ST, minorities, freed bonded labourers, widows, persons with disabilities.",
     "department": "rural_housing",
     "how_to_apply": "Check eligibility on Awaas+ portal (awaassoft.nic.in) or at GP office. Selected from SECC permanent wait list approved by Gram Sabha. Required: Aadhaar, bank account. Geo-tagging at foundation, lintel, and completion stages.",
     "eligibility_questions": [
         {"question": "Do you currently own a pucca (permanent) house?", "eligible_answer": "no"},
         {"question": "Is your household listed in the SECC 2011 / Awaas+ data?", "eligible_answer": "yes"},
         {"question": "Do you live in a rural area?", "eligible_answer": "yes"},
         {"question": "Do you have an Aadhaar card and a bank account?", "eligible_answer": "yes"},
     ]},

    # 5
    {"name": "Bikashita Gaon Bikashita Odisha (BGBO)",
     "description": "State flagship rural infrastructure scheme with Rs. 1,000 crore annual allocation. Covers construction and repair of rural roads, bridges, culverts, school buildings, civic amenities, drainage, and sports facilities. Maximum 35% of funds for road projects. 40% allocation reserved for ITDA (tribal) blocks. Up to 5% for innovative community projects.",
     "eligibility": "All Gram Panchayats in Odisha. Projects must have minimum estimate of Rs. 3 lakh. Proposed by Gram Sabha resolution. Incomplete projects from predecessor scheme (AONO) with 20% expenditure can be completed under BGBO.",
     "department": "infrastructure",
     "how_to_apply": "Projects proposed at Gram Sabha meeting. GP passes resolution and submits to Block office. Technical estimate prepared by Block JE. Approval by District-level committee. Citizens can demand specific projects at Gram Sabha.",
     "eligibility_questions": [
         {"question": "Is the project located within a Gram Panchayat in Odisha?", "eligible_answer": "yes"},
         {"question": "Has a Gram Sabha resolution been passed for this project?", "eligible_answer": "yes"},
         {"question": "Is the project estimate Rs. 3 lakh or more?", "eligible_answer": "yes"},
     ]},

    # 6
    {"name": "Swachh Bharat Mission - Gramin (SBM-G)",
     "description": "National rural sanitation program. Phase II focuses on ODF (Open Defecation Free) sustainability and Solid/Liquid Waste Management (SLWM). Incentive of Rs. 12,000 for Individual Household Latrine (IHHL) construction. Community sanitary complexes, plastic waste management, and faecal sludge management.",
     "eligibility": "Households without toilets (for IHHL incentive). All GPs for community SLWM projects. Priority: BPL households, SC/ST families, women-headed households, persons with disabilities.",
     "department": "sanitation",
     "how_to_apply": "Apply at GP office for IHHL incentive. GP Secretary verifies eligibility. Construct toilet and submit photo evidence. Rs. 12,000 credited to bank account after verification. For SLWM projects, GP passes resolution at Gram Sabha.",
     "eligibility_questions": [
         {"question": "Does your household have an existing toilet?", "eligible_answer": "no"},
         {"question": "Are you residing in a rural area?", "eligible_answer": "yes"},
         {"question": "Are you willing to construct an Individual Household Latrine?", "eligible_answer": "yes"},
     ]},

    # 7
    {"name": "NRLM / Odisha Livelihood Mission (OLM)",
     "description": "National Rural Livelihoods Mission implemented as Odisha Livelihood Mission. Promotes poverty reduction through SHG institution building, financial inclusion, and livelihood enhancement. SHGs receive Revolving Fund (Rs. 15,000) and Community Investment Fund (Rs. 50,000-1,00,000). Bank linkage loans up to Rs. 3+ lakh at subvented interest rates.",
     "eligibility": "Rural BPL women form groups of 10-20 members. Priority: SC/ST households, landless labourers, persons with disabilities, minorities. SHG must maintain regular savings and meetings for 6 months before bank linkage.",
     "department": "rural_livelihoods",
     "how_to_apply": "Contact GP-level Community Resource Person (CRP) or Block OLM Coordinator to join/form an SHG. Required: Aadhaar, bank account for SHG. After 6 months of regular activity, apply for Revolving Fund through Block office.",
     "eligibility_questions": [
         {"question": "Are you a woman residing in a rural area?", "eligible_answer": "yes"},
         {"question": "Does your household fall below the poverty line (BPL)?", "eligible_answer": "yes"},
         {"question": "Are you part of (or willing to form) a Self Help Group of 10-20 members?", "eligible_answer": "yes"},
         {"question": "Has your SHG been meeting and saving regularly for at least 6 months?", "eligible_answer": "yes"},
     ]},

    # 8
    {"name": "15th Finance Commission Grants to PRIs",
     "description": "Central grants to Gram Panchayats, Panchayat Samitis, and Zilla Parishads for basic civic services. 50% tied grants mandatory for drinking water/sanitation (including rainwater harvesting, water recycling). 50% untied grants for any GP-prioritized development. Total allocation for Odisha: Rs. 1,002+ crore for water and sanitation alone.",
     "eligibility": "All three-tier PRIs (Gram Panchayat, Panchayat Samiti, Zilla Parishad) in Odisha. Utilization must be as per approved action plan. Audit and utilization certificates mandatory for next installment.",
     "department": "infrastructure",
     "how_to_apply": "No citizen application needed — grants are released to PRIs automatically. Citizens can influence utilization through Gram Sabha resolutions. Track fund flow at egramswaraj.gov.in. Report misuse to BDO or District Panchayat Officer.",
     "eligibility_questions": [
         {"question": "Is this request related to a Panchayati Raj Institution (GP/Block/ZP)?", "eligible_answer": "yes"},
         {"question": "Has the previous utilization certificate been submitted?", "eligible_answer": "yes"},
         {"question": "Is the proposed use aligned with basic civic services (water, sanitation, roads)?", "eligible_answer": "yes"},
     ]},

    # 9
    {"name": "Nirman Shramik Pucca Ghar Yojana",
     "description": "State housing scheme specifically for registered construction workers. Provides financial assistance for pucca house construction. Implemented through Panchayati Raj Department in convergence with Odisha Building Workers Welfare Board.",
     "eligibility": "Must be registered with Odisha Building and Construction Workers Welfare Board. Must not own a pucca house. Registration must be active (renewed within 3 years).",
     "department": "rural_housing",
     "how_to_apply": "Apply at GP or Block office with construction worker registration certificate, Aadhaar, land ownership/patta document, and bank passbook. GP Secretary verifies and forwards to Block. Sanctioned amount released in installments.",
     "eligibility_questions": [
         {"question": "Are you registered with the Odisha Building and Construction Workers Welfare Board?", "eligible_answer": "yes"},
         {"question": "Is your registration active (renewed within the last 3 years)?", "eligible_answer": "yes"},
         {"question": "Do you currently own a pucca (permanent) house?", "eligible_answer": "no"},
         {"question": "Do you have a land ownership document or patta?", "eligible_answer": "yes"},
     ]},

    # 10
    {"name": "Mission Shakti (SHG Convergence through PR&DW)",
     "description": "Women's SHG empowerment program with convergence through PR&DW Department. SHGs formed under Mission Shakti are linked to MGNREGS works, JJM operations (Jalasathi), GP-level service delivery, and BGBO project monitoring. Interest-free loans up to Rs. 3 lakh for livelihood activities.",
     "eligibility": "Women Self Help Groups registered under Mission Shakti / NRLM. Individual members age 18+ years. SHGs must have minimum 6 months of regular savings and meeting records.",
     "department": "rural_livelihoods",
     "how_to_apply": "Contact Block Mission Shakti / OLM office. SHG registration through CRP at GP level. Required: SHG registration, member Aadhaar cards, joint bank account. Loan applications through bank linkage after grading.",
     "eligibility_questions": [
         {"question": "Is your SHG registered under Mission Shakti or NRLM?", "eligible_answer": "yes"},
         {"question": "Are all members 18 years or older?", "eligible_answer": "yes"},
         {"question": "Has the SHG maintained regular savings and meetings for at least 6 months?", "eligible_answer": "yes"},
     ]},

    # 11 — NEW
    {"name": "Odisha Rural Connectivity Programme (ORCP)",
     "description": "State programme to connect unconnected habitations with 250+ population through all-weather roads. Bridges and causeways included for habitations cut off during monsoon. Convergence with PMGSY-III for upgrading existing rural roads to all-weather standards. Maintenance budget allocated for 5 years post-construction.",
     "eligibility": "Unconnected habitations with population of 250 or more (125 for tribal/hilly areas). Habitation must be listed in the District Rural Road Plan (DRRP). GP resolution required for prioritization.",
     "department": "infrastructure",
     "how_to_apply": "Submit request through GP Sarpanch with village population certificate and GP resolution. Block office verifies habitation eligibility against DRRP. District-level approval by DRDA committee. Track progress at omms.nic.in.",
     "eligibility_questions": [
         {"question": "Is the habitation currently unconnected by an all-weather road?", "eligible_answer": "yes"},
         {"question": "Does the habitation have a population of 250 or more (125 for tribal areas)?", "eligible_answer": "yes"},
         {"question": "Has a GP resolution been passed requesting the road?", "eligible_answer": "yes"},
         {"question": "Is the habitation listed in the District Rural Road Plan?", "eligible_answer": "yes"},
     ]},

    # 12 — NEW
    {"name": "Gopabandhu Gramin Yojana (GGY)",
     "description": "State rural development programme for comprehensive village development in aspirational blocks. Covers drinking water, sanitation, housing, livelihood, health, education, and digital connectivity. Each selected GP receives Rs. 50 lakh over 3 years. 100 most backward GPs selected in Phase I.",
     "eligibility": "Gram Panchayats in aspirational blocks identified by the State Government. Multi-dimensional poverty index used for selection. GP must submit a Village Development Plan approved by Gram Sabha.",
     "department": "panchayati_raj",
     "how_to_apply": "Selected GPs are notified by the District administration. GP prepares a comprehensive Village Development Plan at Gram Sabha. Plan submitted to Block for technical vetting. Funds released in three annual installments against utilization certificates.",
     "eligibility_questions": [
         {"question": "Is your Gram Panchayat located in an aspirational block identified by the State?", "eligible_answer": "yes"},
         {"question": "Has the GP prepared a Village Development Plan?", "eligible_answer": "yes"},
         {"question": "Has the plan been approved by the Gram Sabha?", "eligible_answer": "yes"},
     ]},
]

# ---------------------------------------------------------------------------
# Import function
# ---------------------------------------------------------------------------
async def import_schemes(qdrant) -> int:
    print("\n  Importing government schemes...")
    for i, s in enumerate(SCHEMES):
        print(f"    [{i+1}/{len(SCHEMES)}] {s['name'][:55]}...")
        text = f"{s['name']} {s['description']} {s['eligibility']} {s['how_to_apply']}"
        embedding = await get_embedding(text)
        point = PointStruct(
            id=new_id(), vector=embedding,
            payload={
                "name": s["name"], "description": s["description"],
                "eligibility": s["eligibility"], "department": s["department"],
                "how_to_apply": s["how_to_apply"],
                "eligibility_questions": s.get("eligibility_questions", []),
                "documents": [],
                "created_at": now_utc().isoformat(),
            })
        qdrant.upsert(collection_name="schemes", points=[point], wait=True)
    print(f"  => {len(SCHEMES)} government schemes")
    return len(SCHEMES)
