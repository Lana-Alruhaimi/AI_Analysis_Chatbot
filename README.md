# AI-Powered Review Analysis and Chatbot

This repository contains a three-part pipeline designed to analyze product reviews, perform sentiment analysis using a Large Language Model (LLM), and deploy the results in a real-time data analysis chatbot built with Streamlit.

## Overview
Three sequential steps:

1.  (Data_Analysis_Cleaning.ipynb): Cleans and prepares the raw review data, ensuring text fields are ready for LLM processing and exploding user reviews into individual rows.
2.  (Sentiment_Analysis.py): Uses the Groq API (via the `llama-3.3-70b-versatile` model) to perform sentiment analysis on the cleaned reviews and saves the results.
3.  (Data_Chatbot.py): A Streamlit application that allows users to chat about the analyzed data, offering a choice between the high-speed **Groq Llama 3.3** model and a **local Hugging Face GPT-2** model.

##  Setup and Installation

### Create and Activate Environment

Ensure you have Anaconda or Miniconda installed, then create and activate a new environment:

```bash
conda create -n llm_env python=3.10
conda activate llm_env 

