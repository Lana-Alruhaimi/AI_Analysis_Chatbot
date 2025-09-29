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