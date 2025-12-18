import pandas as pd
import numpy as np
import re

BMAS = [
    "zoledronic acid", "100 ML zoledronic acid 0.04 MG/ML Injection",
    "100 ML zoledronic acid 0.05 MG/ML Injection", "pamidronate", "denosumab",
    "1.7 ML denosumab 70 MG/ML Injection", "denosumab 60 MG/ML Injectable Solution",
    "alendronate", "alendronic acid 70 MG Oral Tablet", "alendronic acid 35 MG Oral Tablet",
    "alendronic acid 10 MG Oral Tablet", "alendronic acid 0.933 MG/ML Oral Solution",
    "ibandronic acid 150 MG Oral Tablet", "ibandronic acid 1 MG/ML Prefilled Syringe",
    "risedronate sodium 150 MG Oral Tablet", "risedronate sodium 35 MG Delayed Release Oral Tablet",
    "risedronate sodium 35 MG Oral Tablet", "risedronate sodium 5 MG Oral Tablet",
    "calcitonin"
]

CHEMOTHERAPY = [
    "0.4 ML methotrexate 50 MG/ML Auto-Injector", "2 ML eribulin mesylate 0.5 MG/ML Injection",
    "20 ML cytarabine 100 MG/ML Injection", "Cytarabine liposome / Daunorubicin Liposomal",
    "arsenic trioxide", "azacitidine", "bendamustine", "bleomycin", "busulfan",
    "busulfan 2 MG Oral Tablet", "capecitabine 150 MG Oral Tablet", "capecitabine 500 MG Oral Tablet",
    "carboplatin", "cisplatin", "cyclophosphamide", "dacarbazine", "daunorubicin", "decitabine",
    "docetaxel", "doxorubicin hydrochloride", "doxorubicin liposome", "eribulin mesylate",
    "etoposide", "fluorouracil", "gemcitabine", "hydroxyurea", "hydroxyurea 500 MG Oral Capsule", "idarubicin",
    "ifosfamide", "irinotecan", "melphalan", "methotrexate", "mitomycin", "mitoxantrone", "oxaliplatin",
    "paclitaxel", "paclitaxel protein-bound", "pemetrexed", "temozolomide", "thiotepa", "trabectedin",
    "vinblastine", "vincristine", "vinorelbine", "pralatrexate", "pegaspargase", "asparaginase Erwinia chrysanthemi",
    "peginterferon alfa-2a"
]

TARGETED_THERAPY = [
    "0.2 ML lanreotide 300 MG/ML Prefilled Syringe", "0.25 ML leuprolide acetate 30 MG/ML Prefilled Syringe",
    "0.375 ML leuprolide acetate 120 MG/ML Prefilled Syringe", "0.375 ML leuprolide acetate 60 MG/ML Prefilled Syringe",
    "0.5 ML lanreotide 240 MG/ML Prefilled Syringe", "0.5 ML leuprolide acetate 60 MG/ML Prefilled Syringe",
    "1 ML hyaluronidase, human recombinant 150 UNT/ML Injection", "1 ML octreotide 0.05 MG/ML Injection",
    "1 ML octreotide 0.1 MG/ML Injection", "1 ML octreotide 0.5 MG/ML Injection", "abiraterone acetate",
    "alemtuzumab", "anastrozole", "atezolizumab", "avelumab", "bevacizumab", "bicalutamide",
    "blinatumomab", "bortezomib", "cabozantinib", "carfilzomib", "cemiplimab", "cetuximab", "copanlisib",
    "daratumumab", "daratumumab / hyaluronidase", "degarelix", "durvalumab", "elotuzumab", "enzalutamide",
    "everolimus", "fulvestrant", "goserelin", "ibrutinib", "imatinib", "ipilimumab", "ixazomib", "lanreotide",
    "lenvatinib", "leuprolide acetate", "luspatercept", "midostaurin", "nivolumab",
    "obinutuzumab", "octreotide", "ofatumumab", "olaparib", "palbociclib", "pembrolizumab",
    "pertuzumab", "polatuzumab vedotin", "ramucirumab", "rituximab", "rituximab / hyaluronidase",
    "ruxolitinib", "sacituzumab govitecan", "sipuleucel-T", "temsirolimus", "trastuzumab",
    "trastuzumab / hyaluronidase", "venetoclax", "letrozole", "letrozole 2.5 MG Oral Tablet",
    "exemestane", "alpelisib", "nintedanib", "sorafenib", "belinostat", "belantamab mafodotin", "ixazomib",
    "sunitinib"
]

def add_drug_features(df_features, df_drug_exposure):
    
    df_out = df_features.copy()
    
    windows = {
        '6m': {'start_days': 270, 'end_days': 120},   
        '12m': {'start_days': 540, 'end_days': 180}   
    }
    
    # Reference date once
    df_out['ref_date'] = df_out['first_bone_event_date'].fillna(df_out['last_of_death_or_visit'])
    
    for suffix, days in windows.items():
        df_out[f'win_start_{suffix}'] = df_out['ref_date'] - pd.Timedelta(days=days['start_days'])
        df_out[f'win_end_{suffix}'] = df_out['ref_date'] - pd.Timedelta(days=days['end_days'])
    
    min_date = min(df_out['win_start_6m'].min(), df_out['win_start_12m'].min())
    max_date = max(df_out['win_end_6m'].max(), df_out['win_end_12m'].max())
    patients = set(df_out['person_id'])
    
    df_drug = df_drug_exposure[
        (df_drug_exposure['person_id'].isin(patients)) &
        (df_drug_exposure['drug_exposure_start_date'].between(min_date, max_date))
    ].copy()
    
    # Target drugs once
    all_drugs = BMAS + CHEMOTHERAPY + TARGETED_THERAPY
    pattern = '|'.join(re.escape(d.lower()) for d in all_drugs)
    df_target = df_drug[
        df_drug['concept_name'].str.lower().str.contains(pattern, na=False, regex=True)
    ].copy()
    
    # Categorize once
    df_target['bmas'] = df_target['concept_name'].str.lower().str.contains(
        '|'.join(re.escape(d.lower()) for d in BMAS), na=False, regex=True
    ).astype(int)
    df_target['chemo'] = df_target['concept_name'].str.lower().str.contains(
        '|'.join(re.escape(d.lower()) for d in CHEMOTHERAPY), na=False, regex=True
    ).astype(int)
    df_target['targeted'] = df_target['concept_name'].str.lower().str.contains(
        '|'.join(re.escape(d.lower()) for d in TARGETED_THERAPY), na=False, regex=True
    ).astype(int)
    
    # Process both windows
    for suffix in ['6m', '12m']:
        # Merge with this window
        df_merged = df_target.merge(
            df_out[['person_id', f'win_start_{suffix}', f'win_end_{suffix}']], 
            on='person_id'
        )
        
        df_valid = df_merged[
            df_merged['drug_exposure_start_date'].between(
                df_merged[f'win_start_{suffix}'], df_merged[f'win_end_{suffix}']
            )
        ]
        
        # any exposure per category
        for cat in ['bmas', 'chemo', 'targeted']:
            exposed = df_valid.groupby('person_id')[cat].any()
            df_out[f'{cat}_{suffix}'] = df_out['person_id'].map(exposed).fillna(0).astype(int)
    
    # Clean up
    drop_cols = ['ref_date'] + [col for col in df_out.columns if col.startswith('win_')]
    return df_out.drop(columns=drop_cols)
