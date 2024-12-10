# google ai imports
import google.generativeai as genai

# custom imports 
from private import GOOGLE_API

# HERE YOU HAVE TO SET UP YOUR CONFIGURATIONS

# MODEL CONFIGS
genai.configure(api_key=GOOGLE_API)
model = genai.GenerativeModel(model_name = "gemini-1.5-flash-002")