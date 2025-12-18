import pandas as pd
import numpy as np
import re

HIGH_FREQ_LABS = [
    "Systolic blood pressure", "Diastolic blood pressure", "Heart rate", "Body temperature",
    "Respiratory rate", "Oxygen saturation measurement", "Body weight", "Peripheral oxygen saturation",
    "Body height", "Platelets [#/volume] in Blood by Automated count", "Creatinine [Mass/volume] in Serum or Plasma",
    "Potassium [Moles/volume] in Serum or Plasma", "Sodium [Moles/volume] in Serum or Plasma",
    "Calcium [Mass/volume] in Serum or Plasma", "Hematocrit [Volume Fraction] of Blood by Automated count",
    "Hemoglobin [Mass/volume] in Blood", "Urea nitrogen [Mass/volume] in Serum or Plasma",
    "Chloride [Moles/volume] in Serum or Plasma", "Carbon dioxide, total [Moles/volume] in Serum or Plasma",
    "Anion gap in Serum or Plasma", "Leukocytes [#/volume] in Blood by Automated count",
    "Erythrocytes [#/volume] in Blood by Automated count", "MCV [Entitic volume] by Automated count",
    "MCH [Entitic mass] by Automated count", "MCHC [Mass/volume] by Automated count",
    "Erythrocyte distribution width [Ratio] by Automated count", "Neutrophils [#/volume] in Blood by Automated count",
    "Immature granulocytes [#/volume] in Blood by Automated count", "Protein [Mass/volume] in Serum or Plasma",
    "Albumin [Mass/volume] in Serum or Plasma", "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
    "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "Bilirubin.total [Mass/volume] in Serum or Plasma",
    "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma", "Monocytes [#/volume] in Blood by Automated count",
    "Lymphocytes [#/volume] in Blood by Automated count", "Eosinophils [#/volume] in Blood by Automated count",
    "Basophils [#/volume] in Blood by Automated count", "Monocytes/100 leukocytes in Blood by Automated count",
    "Lymphocytes/100 leukocytes in Blood by Automated count", "Eosinophils/100 leukocytes in Blood by Automated count",
    "Neutrophils/100 leukocytes in Blood by Automated count", "Basophils/100 leukocytes in Blood by Automated count",
    "Other cells/100 leukocytes in Blood by Automated count", "Immature granulocytes/100 leukocytes in Blood by Automated count",
    "Magnesium [Mass/volume] in Serum or Plasma", "Phosphate [Mass/volume] in Serum or Plasma",
    "Nucleated erythrocytes [#/volume] in Blood by Automated count", "Nucleated erythrocytes/100 leukocytes [Ratio] in Blood by Automated count",
    "Glomerular filtration rate/1.73 sq M.predicted among non-blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)",
    "Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)",
    "Bilirubin.direct [Mass/volume] in Serum or Plasma", "Glucose [Mass/volume] in Capillary blood by Glucometer",
    "Lactate dehydrogenase [Enzymatic activity/volume] in Serum or Plasma by Lactate to pyruvate reaction"
]


def add_measurement_features(df_features, df_measurements):
    
    df_out = df_features.copy()
    df_out['ref_date'] = df_out['first_bone_event_date'].fillna(df_out['last_of_death_or_visit'])
    
    # BOTH windows
    df_out['win_start_6m'] = df_out['ref_date'] - pd.Timedelta(days=270)
    df_out['win_end_6m'] = df_out['ref_date'] - pd.Timedelta(days=120)
    df_out['win_start_12m'] = df_out['ref_date'] - pd.Timedelta(days=540)
    df_out['win_end_12m'] = df_out['ref_date'] - pd.Timedelta(days=180)
    df_out['T_ref'] = pd.to_datetime(df_out['first_bone_event_date'].fillna(df_out['last_of_death_or_visit']))
    df_measurements['measurement_date'] = pd.to_datetime(df_measurements['measurement_date'])
    
    # **CRITICAL: PRE-FILTER to relevant patients/dates FIRST**
    patients = df_out['person_id'].unique()
    min_date = df_out[['win_start_6m', 'win_start_12m']].min().min()
    max_date = df_out[['win_end_6m', 'win_end_12m']].max().max()
    
    df_labs = df_measurements[
        (df_measurements['person_id'].isin(patients)) &
        (df_measurements['measurement_date'].between(min_date, max_date)) &
        (df_measurements['concept_name'].isin(HIGH_FREQ_LABS))  # ONLY your 55 labs!
    ].copy()
    
    print(f"Filtered to {len(df_labs):,} measurements for 55 high-freq labs")
    
    # **ONE broadcast merge** - windows to ALL labs
    df_labs = df_labs.merge(
        df_out[['person_id', 'win_start_6m', 'win_end_6m', 'win_start_12m', 'win_end_12m']],
        on='person_id',
        how='left'
    )
    
    # **VECTORIZED window filtering**
    df_labs_6m = df_labs[
        df_labs['measurement_date'].between(df_labs['win_start_6m'], df_labs['win_end_6m'])
    ].sort_values(['person_id', 'concept_name', 'measurement_date'])
    
    df_labs_12m = df_labs[
        df_labs['measurement_date'].between(df_labs['win_start_12m'], df_labs['win_end_12m'])
    ].sort_values(['person_id', 'concept_name', 'measurement_date'])
    
    print(f"6m window: {len(df_labs_6m):,} records | 12m window: {len(df_labs_12m):,} records")
    
    # **55 labs × 4 features = 220 total features (delta + last for 2 windows)**
    for lab_name in HIGH_FREQ_LABS:
        safe_name = (lab_name.lower()
                    .replace(' ', '_').replace('/', '_').replace('[', '').replace(']', '')
                    .replace('(', '').replace(')', '').replace(',', '')[:25])
    # 6m features
        lab_6m = df_labs_6m[df_labs_6m['concept_name'] == lab_name]
        if len(lab_6m) >= 20:  # Min threshold
            first_6m = lab_6m.groupby('person_id')['value_as_number'].first()
            last_6m = lab_6m.groupby('person_id')['value_as_number'].last()
            delta_6m = last_6m - first_6m
            df_out[f'{safe_name}_delta_6m'] = df_out['person_id'].map(delta_6m).fillna(0)
            df_out[f'{safe_name}_last_6m'] = df_out['person_id'].map(last_6m).fillna(0)
        
        # 12m features  
        lab_12m = df_labs_12m[df_labs_12m['concept_name'] == lab_name]
        if len(lab_12m) >= 20:
            first_12m = lab_12m.groupby('person_id')['value_as_number'].first()
            last_12m = lab_12m.groupby('person_id')['value_as_number'].last()
            delta_12m = last_12m - first_12m
            df_out[f'{safe_name}_delta_12m'] = df_out['person_id'].map(delta_12m).fillna(0)
            df_out[f'{safe_name}_last_12m'] = df_out['person_id'].map(last_12m).fillna(0)
    
    # Cleanup
    drop_cols = ['ref_date'] + [col for col in df_out.columns if col.startswith('win_')]
    return df_out.drop(columns=drop_cols)


