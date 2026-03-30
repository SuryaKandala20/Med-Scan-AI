"""
modules/vector_db.py — ChromaDB Vector Database for Medical Knowledge RAG

What this does:
  1. Stores medical knowledge as vector embeddings in ChromaDB
  2. When the LLM needs to answer about a condition/treatment, 
     we FIRST retrieve relevant knowledge from the vector DB
  3. That retrieved context is injected into the LLM prompt
  4. This is RAG (Retrieval-Augmented Generation) — reduces hallucinations

Collections:
  - medical_conditions: diseases, symptoms, descriptions
  - treatments: care tips, medications, procedures  
  - drug_info: common medications, interactions, side effects
  - emergency_protocols: when to call 911, red flags
  - specialist_directory: which doctor for which condition

Embedding model: all-MiniLM-L6-v2 (free, runs locally, 384-dim vectors)
Vector DB: ChromaDB (free, local, no server needed)
"""

import json
import chromadb
from chromadb.config import Settings
from pathlib import Path

CHROMA_DIR = Path("data/chromadb")
COLLECTION_NAME_CONDITIONS = "medical_conditions"
COLLECTION_NAME_TREATMENTS = "treatments"
COLLECTION_NAME_EMERGENCIES = "emergency_protocols"


# ──────────────────────────────────────────────
# Medical knowledge base (curated, vetted content)
# This is what gets embedded and stored in ChromaDB
# ──────────────────────────────────────────────
MEDICAL_KNOWLEDGE = [
    # ── Respiratory ──
    {
        "id": "cond_common_cold",
        "category": "condition",
        "condition": "Common Cold",
        "text": "The common cold is a viral infection of the upper respiratory tract affecting the nose and throat. Caused by rhinoviruses (most common), coronaviruses, and others. Symptoms include runny nose, sneezing, sore throat, mild cough, low-grade fever, and fatigue. Usually self-limiting, resolving in 7-10 days. Spread through airborne droplets and surface contact. Most contagious in the first 2-3 days.",
        "symptoms": "runny nose, sneezing, sore throat, cough, low fever, fatigue, congestion, post-nasal drip",
        "triage": "Routine",
        "specialist": "General Practitioner",
    },
    {
        "id": "treat_common_cold",
        "category": "treatment",
        "condition": "Common Cold",
        "text": "Treatment for the common cold is supportive — there is no cure. Rest and adequate sleep are essential. Stay hydrated with water, warm broths, and herbal teas. Honey (1 teaspoon) soothes sore throat and cough — not for children under 1 year. Saline nasal spray or neti pot helps congestion. Steam inhalation provides temporary relief. Over-the-counter options: acetaminophen or ibuprofen for pain/fever, pseudoephedrine or phenylephrine for congestion, dextromethorphan for cough. Zinc lozenges within 24 hours of onset may reduce duration. Vitamin C has modest benefit if taken regularly. Antibiotics do NOT work for colds (viral, not bacterial). See a doctor if symptoms worsen after 10 days, fever above 103°F, or difficulty breathing develops.",
    },
    {
        "id": "cond_influenza",
        "category": "condition",
        "condition": "Influenza (Flu)",
        "text": "Influenza is a contagious respiratory illness caused by influenza A or B viruses. More severe than the common cold. Characterized by sudden onset of high fever (101-104°F), severe muscle aches and body pain, extreme fatigue, headache, dry cough, sore throat, and sometimes vomiting/diarrhea. Incubation period 1-4 days. Most contagious in first 3-4 days of illness. Can lead to serious complications including pneumonia, especially in elderly, children, pregnant women, and immunocompromised individuals. Seasonal peaks typically October through March in Northern Hemisphere.",
        "symptoms": "high fever, severe muscle aches, extreme fatigue, headache, dry cough, sore throat, chills, body pain",
        "triage": "Same-day",
        "specialist": "General Practitioner",
    },
    {
        "id": "treat_influenza",
        "category": "treatment",
        "condition": "Influenza (Flu)",
        "text": "Antiviral medications are most effective when started within 48 hours of symptom onset. Oseltamivir (Tamiflu) is the most commonly prescribed — 75mg twice daily for 5 days. Baloxavir (Xofluza) is a single-dose alternative. Rest is critical — the body needs energy to fight the virus. Drink fluids frequently to prevent dehydration — water, electrolyte drinks, warm soup. Over-the-counter acetaminophen or ibuprofen for fever and body aches. Avoid aspirin in children/teens (risk of Reye's syndrome). Isolate from others to prevent spreading — stay home for at least 24 hours after fever resolves. Annual flu vaccination is the best prevention. Seek emergency care for: difficulty breathing, persistent chest pain, confusion, severe vomiting, flu symptoms that improve then return with fever and worse cough.",
    },
    # ── Neurological ──
    {
        "id": "cond_migraine",
        "category": "condition",
        "condition": "Migraine",
        "text": "Migraine is a neurological condition characterized by intense, often one-sided throbbing headache lasting 4-72 hours. Affects approximately 12% of the population, more common in women (3:1 ratio). Four phases: prodrome (mood changes, food cravings 1-2 days before), aura (visual disturbances, numbness in 25% of patients lasting 20-60 minutes), headache phase (intense throbbing pain, nausea, vomiting, light and sound sensitivity), and postdrome (fatigue, difficulty concentrating for 1-2 days after). Common triggers include stress, hormonal changes, certain foods (aged cheese, alcohol, chocolate, MSG), sleep changes, weather changes, strong smells, and bright lights. Classified as episodic (<15 days/month) or chronic (≥15 days/month).",
        "symptoms": "one-sided throbbing headache, nausea, vomiting, light sensitivity, sound sensitivity, visual aura, numbness, fatigue",
        "triage": "Same-day",
        "specialist": "Neurologist",
    },
    {
        "id": "treat_migraine",
        "category": "treatment",
        "condition": "Migraine",
        "text": "Acute treatment (during attack): Rest in a dark, quiet room. Apply cold compress to forehead or back of neck. Over-the-counter NSAIDs (ibuprofen 400-800mg, naproxen 500mg) work best when taken early. Triptans (sumatriptan, rizatriptan) are migraine-specific medications — prescription needed. Anti-nausea medication (ondansetron, metoclopramide) if vomiting. Caffeine can enhance pain relief (small amount). Preventive treatment (if 4+ migraines/month): Beta-blockers (propranolol), antidepressants (amitriptyline), anti-seizure (topiramate, valproate), CGRP monoclonal antibodies (erenumab, fremanezumab — newer, highly effective). Lifestyle: Keep a headache diary to identify triggers. Maintain regular sleep schedule. Regular aerobic exercise (30 min, 5x/week). Stay hydrated. Stress management techniques. Avoid known dietary triggers. Consider magnesium supplementation (400-600mg/day). Biofeedback and cognitive behavioral therapy have evidence for migraine prevention.",
    },
    # ── Cardiovascular ──
    {
        "id": "cond_heart_attack",
        "category": "condition",
        "condition": "Heart Attack (Myocardial Infarction)",
        "text": "A heart attack occurs when blood flow to part of the heart muscle is blocked, usually by a blood clot in a coronary artery. This is a TIME-CRITICAL EMERGENCY. Classic symptoms: crushing chest pain or pressure (described as 'elephant sitting on chest'), pain radiating to left arm, jaw, neck, or back, shortness of breath, cold sweat, nausea, lightheadedness. IMPORTANT: Women may have atypical symptoms — unusual fatigue, back pain, jaw pain, nausea without chest pain. Every minute of delay in treatment increases heart muscle damage. Risk factors: age over 45 (men) or 55 (women), smoking, high blood pressure, high cholesterol, diabetes, family history, obesity, sedentary lifestyle.",
        "symptoms": "crushing chest pain, arm pain radiating, jaw pain, shortness of breath, cold sweat, nausea, lightheadedness, fatigue",
        "triage": "Emergency",
        "specialist": "Emergency Medicine / Cardiologist",
    },
    {
        "id": "emerg_heart_attack",
        "category": "emergency",
        "condition": "Heart Attack",
        "text": "EMERGENCY PROTOCOL FOR HEART ATTACK: 1. CALL 911 IMMEDIATELY — do not wait, do not drive yourself. 2. Chew one regular aspirin (325mg) if not allergic — chewing absorbs faster than swallowing. 3. Sit or lie in a comfortable position. 4. Loosen any tight clothing. 5. If prescribed nitroglycerin, take as directed. 6. Note the EXACT TIME symptoms started — critical for treatment decisions (clot-busting drugs and angioplasty are most effective within the first 90 minutes). 7. If the person becomes unconscious and stops breathing, begin CPR if trained. 8. Do NOT give anything by mouth except aspirin. Treatment at hospital: emergency angioplasty (PCI) to open blocked artery, or thrombolytic drugs to dissolve the clot.",
    },
    {
        "id": "cond_hypertension",
        "category": "condition",
        "condition": "Hypertension (High Blood Pressure)",
        "text": "Hypertension is persistently elevated blood pressure above 130/80 mmHg. Called 'the silent killer' because it often has no symptoms until serious damage occurs. Stages: Elevated (120-129/<80), Stage 1 (130-139/80-89), Stage 2 (≥140/≥90), Hypertensive Crisis (>180/>120 — emergency). Long-term complications: stroke, heart attack, heart failure, kidney disease, vision loss, cognitive decline. Risk factors: age, family history, obesity, high sodium diet, sedentary lifestyle, excess alcohol, smoking, chronic stress, sleep apnea. Secondary causes: kidney disease, thyroid disorders, adrenal tumors, certain medications (NSAIDs, birth control pills, decongestants).",
        "symptoms": "usually none (silent), headache at very high levels, dizziness, vision changes, chest pain, shortness of breath",
        "triage": "Same-day",
        "specialist": "Cardiologist or General Practitioner",
    },
    {
        "id": "treat_hypertension",
        "category": "treatment",
        "condition": "Hypertension",
        "text": "Lifestyle modifications (first-line for all stages): DASH diet — rich in fruits, vegetables, whole grains, lean protein; low in saturated fat and sodium. Reduce sodium to less than 2,300mg/day (ideally 1,500mg). Regular aerobic exercise — 150 minutes/week moderate intensity (brisk walking, cycling). Maintain healthy weight — losing even 5-10 pounds helps. Limit alcohol (≤1 drink/day women, ≤2 men). Quit smoking. Manage stress. Get adequate sleep (7-8 hours). Medications (when lifestyle alone insufficient): ACE inhibitors (lisinopril, enalapril), ARBs (losartan, valsartan), calcium channel blockers (amlodipine), thiazide diuretics (hydrochlorothiazide). Most patients need 2 or more medications. Monitor blood pressure at home — keep a log for your doctor. Hypertensive crisis (>180/120): seek immediate emergency care.",
    },
    # ── Gastrointestinal ──
    {
        "id": "cond_gastroenteritis",
        "category": "condition",
        "condition": "Gastroenteritis (Stomach Flu)",
        "text": "Gastroenteritis is inflammation of the stomach and intestines, usually from viral infection (norovirus most common, rotavirus in children) or bacterial infection (Salmonella, E. coli, Campylobacter from contaminated food). Symptoms: nausea, vomiting, watery diarrhea, abdominal cramps, low-grade fever, headache, muscle aches. Onset typically 1-3 days after exposure. Usually self-limiting, resolving in 1-3 days (viral) or 3-7 days (bacterial). Main danger is dehydration, especially in children, elderly, and immunocompromised. Spread through contaminated food/water, person-to-person contact, and contaminated surfaces.",
        "symptoms": "nausea, vomiting, watery diarrhea, abdominal cramps, low fever, headache, muscle aches, loss of appetite",
        "triage": "Routine",
        "specialist": "General Practitioner or Gastroenterologist",
    },
    {
        "id": "treat_gastroenteritis",
        "category": "treatment",
        "condition": "Gastroenteritis",
        "text": "Primary treatment is preventing dehydration. Oral rehydration: sip small amounts of clear fluids frequently — water, oral rehydration solution (ORS like Pedialyte), clear broths. Avoid gulping large amounts (triggers vomiting). Ice chips or popsicles if unable to keep liquids down. Diet progression: BRAT diet when able to eat (Bananas, Rice, Applesauce, Toast). Gradually reintroduce bland foods. Avoid dairy, caffeine, alcohol, fatty/spicy foods for 48 hours after symptoms resolve. Medications: bismuth subsalicylate (Pepto-Bismol) for mild symptoms. Anti-diarrheal (loperamide/Imodium) — use cautiously, do NOT use if bloody diarrhea or high fever. Ondansetron (Zofran, prescription) for severe vomiting. Antibiotics only for confirmed bacterial cases. Probiotics may shorten illness duration. Seek medical care if: unable to keep fluids down for 24+ hours, bloody stool, fever above 104°F, signs of severe dehydration (dark urine, dizziness, dry mouth, no tears in children).",
    },
    # ── Urinary ──
    {
        "id": "cond_uti",
        "category": "condition",
        "condition": "Urinary Tract Infection (UTI)",
        "text": "UTI is a bacterial infection in any part of the urinary system — most commonly the bladder (cystitis) and urethra (urethritis). Far more common in women due to shorter urethra. Caused primarily by E. coli bacteria. Lower UTI symptoms: burning/painful urination (dysuria), frequent urge to urinate, passing small amounts, cloudy or strong-smelling urine, pelvic pressure, blood in urine (hematuria). Upper UTI (pyelonephritis/kidney infection): fever, chills, back/flank pain, nausea, vomiting — this is more serious. Risk factors: female anatomy, sexual activity, certain birth control (diaphragms, spermicides), menopause, urinary tract abnormalities, catheter use, weakened immune system.",
        "symptoms": "burning urination, frequent urination, urgency, cloudy urine, strong smelling urine, pelvic pain, blood in urine",
        "triage": "Same-day",
        "specialist": "General Practitioner or Urologist",
    },
    # ── Endocrine ──
    {
        "id": "cond_diabetes_t2",
        "category": "condition",
        "condition": "Type 2 Diabetes",
        "text": "Type 2 diabetes is a metabolic disorder where the body becomes resistant to insulin or doesn't produce enough, leading to chronically elevated blood sugar. Develops gradually, often over years. Affects approximately 37 million Americans. Early symptoms: increased thirst (polydipsia), frequent urination (polyuria), increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing wounds, frequent infections, numbness/tingling in hands or feet, areas of darkened skin (acanthosis nigricans). Diagnosis: fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, random glucose ≥200 mg/dL with symptoms. Pre-diabetes: fasting 100-125, HbA1c 5.7-6.4%. Risk factors: overweight/obesity, age >45, family history, sedentary lifestyle, gestational diabetes history, polycystic ovary syndrome.",
        "symptoms": "increased thirst, frequent urination, increased hunger, weight loss, fatigue, blurred vision, slow healing, numbness, tingling, darkened skin",
        "triage": "Same-day",
        "specialist": "Endocrinologist or General Practitioner",
    },
    {
        "id": "cond_hypothyroid",
        "category": "condition",
        "condition": "Hypothyroidism",
        "text": "Hypothyroidism occurs when the thyroid gland doesn't produce enough thyroid hormones (T3 and T4), slowing metabolism throughout the body. Most common cause is Hashimoto's thyroiditis (autoimmune). Affects women 5-8x more than men. Symptoms develop gradually: fatigue, weight gain despite normal eating, cold intolerance, constipation, dry skin, thinning hair, puffy face, hoarse voice, muscle weakness, elevated cholesterol, joint pain/stiffness, irregular or heavy periods, depression, memory problems, slow heart rate. Diagnosis: elevated TSH (primary screening test), low free T4. Subclinical hypothyroidism: elevated TSH with normal T4 — may or may not need treatment.",
        "symptoms": "fatigue, weight gain, cold intolerance, constipation, dry skin, hair loss, puffy face, hoarse voice, depression, memory problems",
        "triage": "Routine",
        "specialist": "Endocrinologist",
    },
    # ── Allergic ──
    {
        "id": "cond_allergy",
        "category": "condition",
        "condition": "Allergic Reaction",
        "text": "Allergic reactions occur when the immune system overreacts to a normally harmless substance (allergen). Severity ranges from mild to life-threatening anaphylaxis. Common allergens: pollen, dust mites, pet dander, mold, certain foods (peanuts, tree nuts, shellfish, milk, eggs, wheat, soy), insect stings, medications (penicillin, NSAIDs), latex. Mild symptoms: hives, itching, rash, nasal congestion, sneezing, watery eyes. Moderate: widespread hives, swelling, abdominal pain, nausea. Severe/anaphylaxis: throat swelling, difficulty breathing, wheezing, rapid pulse, dizziness, blood pressure drop, loss of consciousness. Anaphylaxis can be fatal within minutes if untreated.",
        "symptoms": "hives, itching, rash, swelling, congestion, sneezing, watery eyes, throat tightness, difficulty breathing, dizziness",
        "triage": "Urgent",
        "specialist": "Allergist/Immunologist",
    },
    {
        "id": "emerg_anaphylaxis",
        "category": "emergency",
        "condition": "Anaphylaxis",
        "text": "ANAPHYLAXIS EMERGENCY PROTOCOL: This is a life-threatening allergic reaction. 1. CALL 911 IMMEDIATELY. 2. Use epinephrine auto-injector (EpiPen) if available — inject into outer thigh, can inject through clothing. Adults: 0.3mg, Children <30kg: 0.15mg. 3. Lay the person flat with legs elevated (unless breathing difficulty — then sit them up). 4. A second dose of epinephrine can be given after 5-15 minutes if no improvement. 5. If the person stops breathing, begin CPR. 6. Do NOT give oral medications if person is having trouble swallowing. 7. Even if symptoms improve with epinephrine, go to the ER — biphasic reactions can occur hours later. After emergency: see an allergist for testing, get prescribed EpiPen to carry at all times, consider medical alert bracelet.",
    },
    # ── Mental Health ──
    {
        "id": "cond_anxiety",
        "category": "condition",
        "condition": "Anxiety Disorder",
        "text": "Anxiety disorders are a group of mental health conditions characterized by excessive fear, worry, and related behavioral disturbances. Most common mental illness worldwide, affecting ~30% of adults at some point. Types: Generalized Anxiety Disorder (persistent excessive worry about everyday things), Panic Disorder (recurrent unexpected panic attacks), Social Anxiety Disorder, Specific Phobias. Physical manifestations: rapid heartbeat, chest tightness, shortness of breath, dizziness, trembling, sweating, GI upset, muscle tension, insomnia, fatigue. These physical symptoms often mimic cardiac conditions. Panic attacks: sudden onset of intense fear with physical symptoms peaking in minutes — chest pain, pounding heart, feeling of choking, numbness, derealization. Important: rule out cardiac causes if chest symptoms are present, especially in new-onset symptoms.",
        "symptoms": "excessive worry, rapid heartbeat, chest tightness, shortness of breath, dizziness, sweating, trembling, muscle tension, insomnia, panic attacks",
        "triage": "Routine",
        "specialist": "Psychiatrist or Psychologist",
    },
    {
        "id": "treat_anxiety",
        "category": "treatment",
        "condition": "Anxiety Disorder",
        "text": "First step: rule out medical causes (thyroid, cardiac, medication side effects). Evidence-based treatments: Cognitive Behavioral Therapy (CBT) — most effective therapy, teaches identification and restructuring of anxious thoughts. Exposure therapy for phobias and social anxiety. Medications: SSRIs (sertraline, escitalopram) — first-line, takes 4-6 weeks for full effect. SNRIs (venlafaxine, duloxetine). Buspirone — non-addictive anxiolytic. Benzodiazepines (lorazepam, alprazolam) — short-term use only due to dependence risk. Acute panic attack management: 4-7-8 breathing technique (inhale 4 seconds, hold 7, exhale 8). Grounding techniques (5-4-3-2-1 senses). Progressive muscle relaxation. Lifestyle: regular aerobic exercise (as effective as medication for mild-moderate anxiety), adequate sleep, limit caffeine and alcohol, mindfulness meditation, yoga. Long-term: maintain social connections, develop stress management routine, consider support groups.",
    },
    # ── Musculoskeletal ──
    {
        "id": "cond_rheumatoid",
        "category": "condition",
        "condition": "Rheumatoid Arthritis",
        "text": "Rheumatoid arthritis (RA) is a chronic autoimmune disease where the immune system attacks the lining of joints (synovium), causing inflammation and joint damage. Affects approximately 1% of the population, women 2-3x more than men, typically onset age 30-60. Symptoms: joint pain, swelling, and stiffness (especially morning stiffness lasting >30 minutes), typically symmetrical (both hands, both knees), fatigue, low-grade fever, weight loss. Most commonly affects small joints first (fingers, wrists), then larger joints. Unlike osteoarthritis, RA is worse with rest and improves with movement. Diagnosis: blood tests (RF, anti-CCP, ESR, CRP), imaging (X-rays, ultrasound, MRI). Early aggressive treatment is critical to prevent irreversible joint destruction.",
        "symptoms": "joint pain, joint swelling, morning stiffness, symmetrical joint involvement, fatigue, low fever, weight loss",
        "triage": "Routine",
        "specialist": "Rheumatologist",
    },
    # ── Pneumonia ──
    {
        "id": "cond_pneumonia",
        "category": "condition",
        "condition": "Pneumonia",
        "text": "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Can be caused by bacteria (most commonly Streptococcus pneumoniae), viruses (influenza, COVID-19, RSV), or fungi. Community-acquired pneumonia is the most common form. Symptoms: persistent cough (may produce green, yellow, or bloody mucus), high fever with chills and sweating, shortness of breath even at rest, sharp chest pain that worsens with breathing or coughing, fatigue and weakness, nausea/vomiting/diarrhea, confusion (especially in elderly). Chest X-ray is the primary diagnostic tool. Can be life-threatening, especially in elderly (>65), children (<2), immunocompromised individuals, and those with chronic conditions.",
        "symptoms": "productive cough, high fever, chills, shortness of breath, chest pain with breathing, fatigue, confusion in elderly",
        "triage": "Urgent",
        "specialist": "Pulmonologist or Emergency Medicine",
    },
    # ── Anemia ──
    {
        "id": "cond_anemia",
        "category": "condition",
        "condition": "Anemia",
        "text": "Anemia is a condition where you lack enough healthy red blood cells or hemoglobin to carry adequate oxygen to tissues. Most common blood disorder. Types: iron-deficiency (most common — from blood loss, poor absorption, or inadequate intake), vitamin B12 deficiency, folate deficiency, chronic disease anemia, hemolytic anemia, aplastic anemia, sickle cell disease. Symptoms: fatigue and weakness, pale or yellowish skin, irregular or rapid heartbeat, shortness of breath, dizziness or lightheadedness, cold hands and feet, headache, brittle nails, unusual cravings (pica — ice, dirt, starch). Diagnosis: Complete Blood Count (CBC) showing low hemoglobin (<12 g/dL women, <13.5 g/dL men), low hematocrit, low red blood cell count. Additional tests: iron studies, B12, folate, reticulocyte count, peripheral blood smear.",
        "symptoms": "fatigue, weakness, pale skin, rapid heartbeat, shortness of breath, dizziness, cold extremities, headache, brittle nails",
        "triage": "Same-day",
        "specialist": "Hematologist or General Practitioner",
    },
]


class MedicalVectorDB:
    """ChromaDB-backed medical knowledge retrieval system."""

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = None
        self._initialized = False

    def initialize(self):
        """Create/load the collection and populate with medical knowledge."""
        if self._initialized:
            return

        # Use ChromaDB's default embedding function (all-MiniLM-L6-v2)
        # This downloads the model automatically on first run (~80MB)
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"description": "Medical conditions, treatments, and emergency protocols"},
        )

        # Check if already populated
        if self.collection.count() >= len(MEDICAL_KNOWLEDGE):
            self._initialized = True
            print(f"✅ Vector DB loaded: {self.collection.count()} documents")
            return

        # Populate with medical knowledge
        print("Populating vector DB with medical knowledge...")
        ids = []
        documents = []
        metadatas = []

        for entry in MEDICAL_KNOWLEDGE:
            ids.append(entry["id"])
            documents.append(entry["text"])
            metadata = {
                "category": entry.get("category", ""),
                "condition": entry.get("condition", ""),
                "triage": entry.get("triage", ""),
                "specialist": entry.get("specialist", ""),
            }
            # ChromaDB metadata values must be str, int, float, or bool
            metadatas.append({k: v for k, v in metadata.items() if v})

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        self._initialized = True
        print(f"✅ Vector DB populated: {self.collection.count()} documents")

    def query(self, query_text: str, n_results: int = 5, category: str = None) -> list:
        """
        Search the medical knowledge base.

        Args:
            query_text: natural language query (e.g., "headache with nausea and light sensitivity")
            n_results: number of results to return
            category: filter by category ('condition', 'treatment', 'emergency')

        Returns:
            list of dicts with: id, text, condition, category, distance
        """
        if not self._initialized:
            self.initialize()

        where_filter = None
        if category:
            where_filter = {"category": category}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
        )

        output = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else None
                output.append({
                    "id": results["ids"][0][i],
                    "text": doc,
                    "condition": meta.get("condition", ""),
                    "category": meta.get("category", ""),
                    "specialist": meta.get("specialist", ""),
                    "triage": meta.get("triage", ""),
                    "distance": dist,
                })
        return output

    def get_context_for_symptoms(self, symptoms_text: str) -> str:
        """
        Get relevant medical context for a set of symptoms.
        This context is injected into the LLM prompt for RAG.

        Returns a formatted string of relevant medical knowledge.
        """
        # Query for conditions matching symptoms
        conditions = self.query(symptoms_text, n_results=3, category="condition")

        # Query for treatments
        treatments = self.query(symptoms_text, n_results=2, category="treatment")

        # Query for emergency protocols if any
        emergencies = self.query(symptoms_text, n_results=1, category="emergency")

        context_parts = []

        if conditions:
            context_parts.append("=== RELEVANT CONDITIONS ===")
            for c in conditions:
                context_parts.append(f"[{c['condition']}] (Triage: {c['triage']}, Specialist: {c['specialist']})")
                context_parts.append(c["text"])
                context_parts.append("")

        if treatments:
            context_parts.append("=== TREATMENT INFORMATION ===")
            for t in treatments:
                context_parts.append(f"[{t['condition']} Treatment]")
                context_parts.append(t["text"])
                context_parts.append("")

        if emergencies and emergencies[0].get("distance", 999) < 1.5:
            context_parts.append("=== EMERGENCY PROTOCOLS ===")
            for e in emergencies:
                context_parts.append(e["text"])
                context_parts.append("")

        return "\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get vector DB statistics."""
        if not self._initialized:
            self.initialize()
        return {
            "total_documents": self.collection.count(),
            "categories": {
                "conditions": len([d for d in MEDICAL_KNOWLEDGE if d.get("category") == "condition"]),
                "treatments": len([d for d in MEDICAL_KNOWLEDGE if d.get("category") == "treatment"]),
                "emergencies": len([d for d in MEDICAL_KNOWLEDGE if d.get("category") == "emergency"]),
            },
            "storage_path": str(CHROMA_DIR),
        }
