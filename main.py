import pandas as pd
import os
from dotenv import load_dotenv
from src.load_data import load_data
from models.six_month_lr import six_month_lr
from models.twelve_month_lr import twelve_month_lr
from models.six_month_rf import six_month_rf
from models.twelve_month_rf import twelve_month_rf

load_dotenv()

FOLDER_PATH = os.getenv("FOLDER_PATH")

df_features = load_data(FOLDER_PATH)

six_month_lr(df_features)
twelve_month_lr(df_features)
six_month_rf(df_features)
twelve_month_rf(df_features)