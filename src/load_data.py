import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
from src.make_dataframe import make_dataframe
from src.add_measurement_features import add_measurement_features
from src.add_drug_features import add_drug_features

def load_data(FOLDER_PATH):

    def path_for(name):
        p = name if name.endswith('.parquet') else f"{name}.parquet"
        return Path.home() / FOLDER_PATH / p

    def load_dd(name):
        return dd.read_parquet(path_for(name), engine='pyarrow')

    def load_frame(name):
        return load_dd(name).compute()
    
    print('Loading files...')

    try:
        df_person = load_frame('person')
        df_death = load_frame('death')
        df_measurement = load_frame('measurement')
        df_drug_exposure = load_frame('drug_exposure')
        df_condition_occurrence = load_frame('condition_occurrence')
        df_procedure_occurrence = load_frame('procedure_occurence')
        df_visit_occurrence = load_frame('visit_occurence')
        print('Done. Dataframes available: df_person, df_death, df_measurement, df_drug_exposure, df_condition_occurrence, df_procedure_occurrence, df_visit_occurrence')
    except Exception as e:
        print('Error when loading — check FOLDER_PATH and that parquet files exist:', e)

    df_features = make_dataframe(df_person, df_death, df_measurement, df_drug_exposure, df_condition_occurrence, df_procedure_occurrence, df_visit_occurrence)
    df_features = add_measurement_features(df_features, df_measurement)
    df_features = add_drug_features(df_features, df_drug_exposure)
    return df_features


