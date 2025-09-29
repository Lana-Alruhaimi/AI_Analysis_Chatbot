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