import os
from groq import Groq

try:
    client = Groq() #looks for the GROQ_API_KEY set in prompt
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    print("Please ensure your GROQ_API_KEY is correctly set.")
    exit()

# 2. Define the completion request
try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain in one sentence why the sky is blue",
            }
        ],
        model="llama-3.3-70b-versatile", 
    )

    # 3. Print the model's response
    print("\n--- Groq API Test Successful ---")
    print(f"Model Used: {chat_completion.model}")
    print("\nModel Response:")
    print(chat_completion.choices[0].message.content)
    print("--------------------------------\n")

except Exception as e:
    # This will catch the Invalid API Key error if the 'set' command failed.
    print(f"\n--- API Request Failed ---")
    print(f"Error Details: {e}")
    print("Please double-check your API key and ensure it's active.")