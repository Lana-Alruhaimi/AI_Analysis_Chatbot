import pandas as pd
from groq import Groq
import plotly.express as px
import os

## File paths/directories
Input_Dir = os.path.join('..', 'Data')
Input_Path = os.path.join(Input_Dir, 'cleaned_sales_data.pkl')
Output_Path = os.path.join(Input_Dir, 'analyzed_sales_data.pkl')

print(f"Data file expected at: {Input_Path}") #not excel because this is the cleaned data

## Load data
try:
    df = pd.read_pickle(Input_Path)
    print(f"Successfully loaded {len(df)} records from 'cleaned_sales_data.pkl'.")
except FileNotFoundError:
    print(f"Error: '{Input_Path}' not found. Run (Data_Analysis_Cleaning.ipynb) first.")
    exit()


## Load pickle
df = None #to initialize as none, and prevent crashing in the future
try:
    df = pd.read_pickle(Input_Path)
    print (f"Successfully loaded {len(df)} records from 'cleaned_review_data.pkl'.")
except FileNotFoundError:
    print(f"ERROR: '{Input_Path}' not found. Make sure (Data_Analysis_Cleaning.ipynb) ran first.")

if df is None:
    print("Cannot do sentiment analysis due to missing data")
else:
    New_Col = 'Customer_Feedback' #engineered in (Data_Analysis_Cleaning.ipynb) 
    print(f"Using new column: {New_Col}")  


## Groq setup
# load API Key from OS and set via command line

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY environment variable not found. Please set it with your terminal (in depth explaination in README file).")
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print(f"Groq Client Initialized (Model: {GROQ_MODEL}).")
    except Exception as e:
        print(f"Error: {e}")