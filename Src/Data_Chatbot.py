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
    
LOCAL_GENERATOR = load_local_generator()


## GROQ response generation
def generate_groq_response(prompt, data_context): #Generate response with Groq API Llama 3.3
    if not GROQ_CLIENT:
        return "ERROR: Groq client not initialized. Check your GROQ_API_KEY."

    system_prompt = (
        "You are an expert data analyst and customer support bot. Your goal is to answer questions about the provided product review data. The data is structured as a Python dictionary. Do not perform any code execution. Keep answers concise and informative. If the answer is not in the data, state it clearly."
        f"\n\n--- DATA CONTEXT ---\n{data_context}")
    
    try:
        chat_completion = GROQ_CLIENT.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"ERROR during Groq generation: {e}"