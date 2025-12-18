# caia-fh-model

# To Run:

Set up variable FOLDER_NAME in .env file as a path to parquet files

Use pyenv to set Python version to 3.10.14

Make the script executable in the terminal using chmod +x run.sh

Run ./run.sh

# Notes:

6924 patients
Event rate: 11.1%
Event defined as bone fracture assessment + radiation therapy after (750-800 patients)

EVENT_IDS = [2110698, 2110700, 2110701, 2110699, 2110696, 2110697, 
             2768451, 2103473, 2103475, 2104914, 2105150, 
             46257752, 46257753, 46257748, 2769730, 2765699]

Only 60 patients had events defined by these ids

We use event date or censor date (last visit or death)

For time related features 

6 month prediction model we only look at data from 8-4 month window before event date or censor date

12 month prediction model we only look at data from 15-9 month window before event date or censor date

Drug exposure features: (patient had any exposure to a drug in the category in the window)

Patient rate of exposure 
Catagory | 6 month | 12 month
BMAS: 5.6% | 6.7%
Chemo: 2.8%| 3.0%
Targeted therapy: 6.9% | 9.2%

Measurement features:

