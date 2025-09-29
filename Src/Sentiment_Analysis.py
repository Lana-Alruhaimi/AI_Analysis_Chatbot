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

    # Prompting
    SYSTEM_PROMPT = ("You are a sentiment classifier, analyze the given customer reviews. Your responses should be only one word, and it should be one of these three: 'POSITIVE', 'NEGATIVE', 'NEUTRAL'. Do not include any other text or symbols. Do not change the capitialization")
    def Get_Senti_Groq(review_text): #calls API to get sentiment label
        if not GROQ_API_KEY:
            return"ERROR_KEY_MISSING"
        
        try:
            chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": review_text}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.01,
                    max_tokens=5 # because we only need 1 word
                ) 
            #Extract and clean
            label = chat_completion.choices[0].message.content.upper().strip()
            if label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                return label
            return "NEUTRAL" # Default if classification is ambiguous
        except Exception as e:
                # print(f"API Error for review: {e}") # Uncomment for debugging
                return "API_ERROR"
        
    ## Running sentiment analysis
    print("Running Sentiment Analysis...")
    Senti_Labels = []
    texts_to_analyze = df[New_Col].astype(str).tolist()

    for i, text in enumerate(texts_to_analyze):
        if i % 10 == 0: # progress indicator every 10 reviews
            print(f"Processing review {i+1} of {len(texts_to_analyze)}...")

        label = Get_Senti_Groq(text)
        Senti_Labels.append(label)


## Display results
df['Sentiment_Label'] = Senti_Labels
print("\n\nAnalysis complete. Displaying first 10 results:\n")
display_cols = ['product_name', New_Col, 'Sentiment_Label']
print(df[display_cols].head(10).to_string())


## Save Results
df.to_pickle(Output_Path)
print(f"\nSentiment analysis results successfully saved to '{Output_Path}'.")
            