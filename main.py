import os
from dotenv import load_dotenv
from src.load_data import load_data

load_dotenv()

FOLDER_PATH = os.getenv("FOLDER_PATH")

load_data(FOLDER_PATH)

