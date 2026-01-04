import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv("Backend/.env")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: No API Key found.")
else:
    genai.configure(api_key=api_key)
    print("Listing available models:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
