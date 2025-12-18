import pandas as pd
import numpy as np

EVENT_IDS = [2110698, 2110700, 2110701, 2110699, 2110696, 2110697, 
             2768451, 2103473, 2103475, 2104914, 2105150, 
             46257752, 46257753, 46257748, 2769730, 2765699]

def make_dataframe(df_person, df_death, df_measurement, df_drug_exposure, df_condition_occurrence, df_procedure_occurrence, df_visit_occurrence):
    # Define date of last visit
    df_last_visit = df_visit_occurrence.groupby('person_id')['visit_end_date'].max().reset_index()
    df_last_visit.rename(columns={'visit_end_date': 'last_activity_date'}, inplace=True)
    df_last_visit['last_activity_date'] = pd.to_datetime(df_last_visit['last_activity_date'], errors='coerce')

    # Define basic person info
    df_features = df_person[['person_id', 'gender_concept_name', 'year_of_birth']].copy()
    df_features['age'] = 2025 - df_features['year_of_birth']

    # Add death date
    df_features = df_features.merge(df_death[['person_id', 'death_date']], on='person_id', how='left')
    df_features = df_features.merge(df_last_visit, on='person_id', how='left')

    df_features['death_date'] = pd.to_datetime(df_features['death_date'], errors='coerce')
    df_features['last_of_death_or_visit'] = np.where(
        df_features['death_date'].notna(),
        df_features['death_date'],
        df_features['last_activity_date'] 
    )

    df_features.dropna(subset=['last_of_death_or_visit'], inplace=True) # DROPS 164 PATIENTS HERE

    # Cross-reference approach: patients with fracture assessment AND radiation therapy after it
    bone_fracture_procedures = df_procedure_occurrence[
        df_procedure_occurrence['concept_name'].str.contains('pathologic fracture|bone fracture|vertebral fracture', case=False, na=False)
    ].copy()
    bone_fracture_procedures['event_date'] = pd.to_datetime(bone_fracture_procedures['procedure_date'], errors='coerce')
    
    radiation_procedures = df_procedure_occurrence[
        df_procedure_occurrence['concept_name'].str.contains('radiation|radiotherapy', case=False, na=False)
    ].copy()
    radiation_procedures['event_date'] = pd.to_datetime(radiation_procedures['procedure_date'], errors='coerce')
    
    # Find patients with fracture AND some radiation after the fracture
    fracture_plus_radiation_patients = set()
    for patient in set(bone_fracture_procedures['person_id']):
        if patient in set(radiation_procedures['person_id']):
            patient_fractures = bone_fracture_procedures[bone_fracture_procedures['person_id'] == patient]['event_date']
            patient_radiation = radiation_procedures[radiation_procedures['person_id'] == patient]['event_date']
            
            earliest_fracture = patient_fractures.min()
            # Check if ANY radiation occurs after the fracture
            if any(rad_date > earliest_fracture for rad_date in patient_radiation):
                fracture_plus_radiation_patients.add(patient)
    
    # Also include patients with original EVENT_IDS
    event_id_procedures = df_procedure_occurrence[df_procedure_occurrence['procedure_concept_id'].isin(EVENT_IDS)].copy()
    event_id_patients = set(event_id_procedures['person_id'])
    
    # Combine both groups of event patients
    all_event_patients = fracture_plus_radiation_patients.union(event_id_patients)
    
    if len(all_event_patients) > 0:
        # For cross-reference patients: use fracture assessment dates
        cross_ref_procedures = bone_fracture_procedures[
            bone_fracture_procedures['person_id'].isin(fracture_plus_radiation_patients)
        ].copy()
        
        # Add EVENT_IDS procedures
        event_id_procedures['event_date'] = pd.to_datetime(event_id_procedures['procedure_date'], errors='coerce')
        
        # Combine all event procedures
        df_proc_mets = pd.concat([cross_ref_procedures, event_id_procedures]).drop_duplicates()
    else:
        df_proc_mets = pd.DataFrame(columns=['person_id', 'procedure_date'])

    df_earliest_bone_event = df_proc_mets.groupby('person_id')['event_date'].min().reset_index()
    df_earliest_bone_event.rename(columns={'event_date': 'first_bone_event_date'}, inplace=True)

    # Merge event date into the feature table
    df_features = df_features.merge(df_earliest_bone_event, on='person_id', how='left')
    df_features['first_bone_event_date'] = pd.to_datetime(df_features['first_bone_event_date'], errors='coerce')

    df_features['T_ref'] = np.where(
        df_features['first_bone_event_date'].notna(),
        df_features['first_bone_event_date'],
        df_features['last_of_death_or_visit']
    )

    # Define Event Status: 1 if bone event occurred, 0 otherwise (Censored)
    df_features['event_status'] = df_features['first_bone_event_date'].notna().astype(int)

    return df_features