import streamlit as st
import pandas as pd
import os
from groq import Groq
from transformers import pipeline, set_seed

## File paths/directories
Input_Dir = os.path.join('.', 'Data')
Data_path = os.path.join(Input_Dir, 'analyzed_sales_data.pkl')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true' # So hugging face doesn't send unneeded msg.s

## LLM info
GROQ_MODEL = "llama-3.3-70b-versatile"
HF_MODEL_NAME = "gpt2" #openai-community/gpt2


## Load data
@st.cache_data # avoid unneeded computations
def load_data(): #loads analyzed DataFrame and terminates the app if the file is missing
    if not os.path.exist(Data_path):
        st.error(f"ERROR: file not found: {Data_path}")
        st.warning ("Make sure you have ran (Sentiment_Analysis.py) first")
        return None
    
    try:
        df = pd.read_pickle(Data_path)
        return df
    except Exception as e:
        st.error(f"ERROR: {e}")
        return None
    
## Initialization (GROQ)
GROQ_CLIENT = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    try:
        GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.warning(f"ERROR in initializing Groq client: {e}. Groq option will be disabled.")
        GROQ_API_KEY = None # disable groq if error
else:
    st.warning("GROQ_API_KEY not found in environment variables. Groq model option will be disabled.")


## Initialization (Hugging Face)
@st.cache_data
def load_local_generator(): #load local Hugging Face GPT-2 generator
    try: 
        generator = pipeline('text-generation', model=HF_MODEL_NAME, device=-1)
        set_seed(42) # For reproducible local generation
        return generator
    except Exception as e:
        st.error(f"ERROR Could not load local Hugging Face model ({HF_MODEL_NAME}): {e}")
        return None
