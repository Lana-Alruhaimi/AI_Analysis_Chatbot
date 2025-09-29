# AI-Powered Review Analysis and Chatbot

This repository contains a three-part pipeline designed to analyze product reviews, perform sentiment analysis using a Large Language Model (LLM), and deploy the results in a real-time data analysis chatbot built with Streamlit.

## Table of Contents:
- Visual Demo
- Overview
- Technology Stack
- Author and Acknowledgement
- Setup and Installation

## Visual Demo
![Chatbot visual demo](https://github.com/Lana-Alruhaimi/AI_Analysis_Chatbot/blob/main/Visual_Demo_Chatbot.gif)

## Overview
Three sequential steps:

1.  (Data_Analysis_Cleaning.ipynb): Cleans and prepares the raw review data, ensuring text fields are ready for LLM processing and exploding user reviews into individual rows.
2.  (Sentiment_Analysis.py): Uses the Groq API (via the `llama-3.3-70b-versatile` model) to perform sentiment analysis on the cleaned reviews and saves the results.
3.  (Data_Chatbot.py): A Streamlit application that allows users to chat about the analyzed data, offering a choice between the high-speed **Groq Llama 3.3** model and a **local Hugging Face GPT-2** model.

## Technology Stack
Language used is Python (libraries used: pandas, openpyxl, groq, streamlit, transformers, torch)

## Author and Acknowledgement
Code written by Lana, Data provided by Amazon

##  Setup and Installation

Ensure you have Anaconda or Miniconda installed, then create and activate a new environment, you also need the Groq API Key to access the Llama 3.3 model. You must set this key as an environment variable before running the scripts.:

```bash
#CONDA ENV:
conda create -n llm_env python=3.10
conda activate llm_env


#GROQ API KEY:
#(replace "YOUR_API_KEY_HERE" with your actual key.)
set GROQ_API_KEY="YOUR_API_KEY_HERE"


#RUNNING SCRIPTS (in order):
python Data_Analysis_Cleaning.py
python Sentiment_Analysis.py
streamlit run 3_Data_Chatbot.py


