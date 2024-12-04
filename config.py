# google ai imports
import google.generativeai as genai

# custom imports 
from private import GOOGLE_API

# HERE YOU HAVE TO SET UP YOUR CONFIGURATIONS

# MODEL CONFIGS

genai.configure(api_key=GOOGLE_API)

model = genai.GenerativeModel(model_name = "gemini-1.5-flash-002", 
                              system_instruction="""You are Visualization Expert, you are always given the list of columns,
                                                    Among all available columns you have to choose 2 that are going to result in the best visualization
                                                    Keep in mind that you DO NOT have to explain something, just return 2 columns from those that were given to you originally!"""
                             )