'''
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModel

liver_disease_flag = [
    "Liver Cirrhosis",
    "Cirrhotic",
    "Chronic liver disease",
    "Fatty Liver",    # optional
    "Hepatic"         # optional, may be too generic
]

# 2️⃣ Portal hypertension / complications
portal_hypertension_flag = [
    "Portal Hypertension",
    "Esophageal Varices",
    "Gastroesophageal varices",
    "Portal Vein Thrombosis",
    "Hepatic ascites"
]

# 3️⃣ Liver tumor / neoplasm
liver_tumor_flag = [
    "Liver carcinoma",
    "Lesion of liver",
    "Liver mass",
    "Neoplasms",
    "Malignant Neoplasms",
    "Neoplastic",
    "Liver and Intrahepatic Biliary Tract Carcinoma",
    "Primary Malignant Liver Neoplasm",
    "Hepatocellular"
]

# 4️⃣ Biliary / duct issues
biliary_flag = [
    "Congenital Biliary Dilatation",
    "biliary dilatation",
    "Intrahepatic bile duct dilatation",
    "Obstruction of biliary tree",
    "Bile Ducts, Extrahepatic"
]

# 5️⃣ Symptoms / general clinical signs
symptoms_flag = [
    "Ascites",       # also in liver disease
    "Splenomegaly",
    "Abdominal Pain",
    "Weight Loss",
    "Fatigue"
]


model_name = "OpenMed/OpenMed-NER-PathologyDetect-PubMed-109M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
ner = pipeline("token-classification", model=model_name, tokenizer=tokenizer)

def split_chunks(clinical_notes: str, chunk_size=512, overlap=50):
    tokens = tokenizer.encode(clinical_notes, add_special_tokens=False)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return [tokenizer.decode(chunk) for chunk in chunks]


def extract_symptoms_from_clinical_notes(clinical_notes: str):

    chunks = split_chunks(clinical_notes)

    all_results = []
    batch_size = 8

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        results = ner(batch)
        all_results.extend(results)

    extracted_symptoms = set()
    for chunk_result in all_results:
        for ent in chunk_result:
            if ent["entity_group"] == "Disease":
                word = ent["word"].replace("##", "").strip()
                extracted_symptoms.add(word)

    return list(extracted_symptoms)


def generate_ner_flags(clinical_notes: str) -> list:
    """
    Takes clinical notes as input, runs NER to extract symptoms/disease mentions,
    and generates a binary list of flags for liver disease, portal hypertension,
    liver tumor, biliary issues, and general symptoms.

    Returns:
        ner_list (list): [liver_disease, portal_hypertension, liver_tumor, biliary, symptoms]
    """
    # 1️⃣ Extract entities from clinical notes
    symptoms = extract_symptoms_from_clinical_notes(clinical_notes)

    # 2️⃣ Define flags (keyword lists)
    flags = [
        liver_tumor_flag,
        liver_disease_flag,
        portal_hypertension_flag,
        biliary_flag,
        symptoms_flag
    ]

    # 3️⃣ Initialize NER binary list
    ner_list = [0] * len(flags)

    # 4️⃣ Check intersection between extracted symptoms and keyword lists
    symptoms_set = {s.lower().strip() for s in symptoms}  # normalize

    for i, flag_keywords in enumerate(flags):
        flag_set = {f.lower().strip() for f in flag_keywords}
        if symptoms_set & flag_set:
            ner_list[i] = 1

    return ner_list
    '''

import random

def generate_ner_flags(dummy_clinical_notes) -> list:
    """
    Mimics the output of the NER pipeline by generating a random binary list of flags.
    The order of flags is the same as the real NER function:
    [liver_tumor, liver_disease, portal_hypertension, biliary, symptoms]
    """
    
    # Generate 0 or 1 for each flag randomly
    return [random.randint(0, 1) for _ in range(5)]