import streamlit as st
import pandas as pd
import os
from groq import Groq
from transformers import pipeline, set_seed

## File paths/directories
Input_Dir = os.path.join('..', 'Data')
Data_path = os.path.join(Input_Dir, 'analyzed_sales_data.pkl')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true' # So hugging face doesn't send unneeded msg.s

## LLM info
GROQ_MODEL = "llama-3.3-70b-versatile"
HF_MODEL_NAME = "gpt2" #openai-community/gpt2


## Load data
@st.cache_data # avoid unneeded computations
def load_data(): #loads analyzed DataFrame and terminates the app if the file is missing
    if not os.path.exists(Data_path):
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


## GPT2 response generation
def generate_local_response(prompt): #Generate response with local Hugging Face GPT2
    if not LOCAL_GENERATOR:
        return "ERROR: Local GPT-2 generator failed to load."

    try:
        # GPT-2 is not designed for instruction following, so we just prime it.
        response = LOCAL_GENERATOR(
            f"User asked about the data: {prompt} The answer is:",
            max_length=150,
            num_return_sequences=1,
            pad_token_id=LOCAL_GENERATOR.tokenizer.eos_token_id
        )
        # Extract and clean the generated text
        text = response[0]['generated_text'].split("The answer is:")[-1].strip()
        return f"(Local GPT-2 Response Less accurate on data queries) {text}"
    except Exception as e:
        return f"Local Generation Error: {e}"
    
## Streamlit

#Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    st.set_page_config(page_title="Data Review Chatbot", layout="centered")
    st.title("Model Selector")

    # Load data and check for errors
    df = load_data()
    if df is None:
        st.stop() 

    # Sidebar creation
    with st.sidebar:
        st.header("Model Configuration")
        
        # Models (based on API Key availability)
        available_models = ["Local GPT-2 (Hugging Face)"]
        if GROQ_API_KEY and GROQ_CLIENT:
            available_models.insert(0, "Groq Llama 3.3 (Fast API)")

        # Selectbox for model choice 
        st.session_state.model_choice = st.selectbox(
            "Select LLM",
            options=available_models
        )
        st.info(f"Currently selected model: **{st.session_state.model_choice}**")
        
        # Data Context
        st.subheader("Data Check")
        st.write(f"Data loaded successfully: **{len(df)} records**")

    st.dataframe(df.head(5)[['product_name', 'Sentiment_Label', 'Customer_Feedback']], use_container_width=True)

    # Chat Interaction
    # display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input handling
    if prompt := st.chat_input("Questions about the data?"):
        st.session_state.messages.append({"role": "user", "content": prompt}) # Add user message to chat history
        with st.chat_message("user"):
            st.markdown(prompt)

        # Response gen
        with st.chat_message("assistant"):
            with st.spinner(f"Generating response with {st.session_state.model_choice}..."):
                data_sample = df[['product_name', 'Sentiment_Label', 'Customer_Feedback']].head(5).to_dict('records') # create context & send sample to LLM
            
                if st.session_state.model_choice == "Groq Llama 3.3 (Fast API)":
                    full_context = f"A sample of the data reviews is: {data_sample}. The dataframe columns are: {list(df.columns)}. Focus only on data points related to product_name, Sentiment_Label, and Customer_Feedback."
                    response = generate_groq_response(prompt, full_context)
                
                elif st.session_state.model_choice == "Local GPT-2 (Hugging Face)":
                    response = generate_local_response(prompt)  # GPT 2 doesn't handle complex context well so we only pass the prompt
                
                else:
                    response = "Model selection error."
                
                st.markdown(response)



if __name__ == "__main__":
    main()

