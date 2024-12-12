# google ai imports
import google.generativeai as genai

# custom imports 
from private import GOOGLE_API

# MODELS CONFIGURATIONS
genai.configure(api_key=GOOGLE_API)
model = genai.GenerativeModel(model_name = "gemini-1.5-flash-002")

evaluation_model = genai.GenerativeModel(model_name="gemini-1.5-flash-002", 
                                         system_instruction=("You are visualization expert who grades the visualization",
                                                             "You are always given the image of the visualization and list of questions",
                                                             "You only have to answer the provided questions with 'yes' or 'no'",
                                                             "When you answer 'yes' you have to return 1 if 'no' then return 0",
                                                             "When it's not clear whether answer 'yes' or 'no', return 0.5",
                                                             "Your answer should be the python list with 1s, 0.5s and 0s according to the questions",
                                                             "You Should not provide any justifications or explanations or complains"))


# QUESTIONS ABOUT VISUALIZATION

# Expressiveness Questions
expressiveness_questions = [
    "Does the visualization encode all the necessary data without omitting key information?",
    "Does the chart avoid including irrelevant or distracting data?",
    "Are the visual encodings (e.g., position, colour, size) directly tied to the data being explored?",
    "Does the chart avoid unnecessary or overly decorative elements?",
    "Are the visual encodings appropriate for the type of data (e.g., quantitative, categorical)?",
    "Are there no ambiguities in how the data is presented?",
    "Does the chart avoid redundantly displaying the same data in multiple encodings?",
    "Does the visualization avoid clutter by excluding unnecessary labels, gridlines, or annotations?",
    "Are the limits of each variable (e.g., maximum, minimum) correctly represented in the visualization?"
]

# Efficiency Questions
efficiency_questions = [
    "Can the main trends or patterns be understood within a few seconds?",
    "Is the chart easy to interpret without requiring prior domain knowledge?",
    "Are the axes and labels concise and self-explanatory?",
    "Are the data relationships apparent at first glance?",
    "Does the chart avoid requiring additional explanation or reference materials?",
    "Does the visualization minimize cognitive load by keeping the design simple?",
    "Is the choice of chart type appropriate for the data exploration task?",
    "Are distracting visual elements (e.g., excessive annotations, gridlines) avoided?"
]

# Engagement Questions
engagement_questions = [
    "Is the visualization visually appealing without compromising clarity?",
    "Does the design make you want to spend more time analyzing the chart?",
    "Are colour schemes used in a way that makes the chart interesting and inviting?",
    "Is the chart easy to share or present to others?",
    "Does the visualization tell a clear, compelling story?",
    "Is the chart memorable or likely to leave an impression after viewing?"
]

# Data-Ink Ratio Questions
data_ink_ratio_questions = [
    "Does the chart avoid excessive use of non-data ink (e.g., borders, shading, or decorations)?",
    "Is the ratio of data ink to non-data ink maximized?",
    "Does the chart avoid excessive use of colour gradients or textures?",
    "Are the gridlines minimized or appropriately styled?",
    "Is the background plain or minimally styled to avoid distractions?",
    "Are legends and labels succinct yet informative?",
    "Are unnecessary chart elements (e.g., 3D effects, shadows) avoided?",
    "Is all 'ink' on the chart directly tied to data representation?",
    "Does the chart avoid redundant encodings that don’t add value?"
]

# Gestalt Principles Questions
gestalt_principles_questions = [
    "Are related data points visually grouped together (proximity)?",
    "Are similar data points represented with similar visual encodings (similarity)?",
    "Are data relationships shown in a way that forms clear patterns (continuity)?",
    "Are overlapping or intersecting elements easy to distinguish (closure)?",
    "Is the alignment of visual elements consistent across the chart?",
    "Are outliers or anomalies easy to separate from clusters of data?",
    "Does the chart avoid creating visual noise from misaligned or scattered points?",
    "Are the axes and data labels properly aligned and spaced?",
    "Does the visual design emphasize important areas of the chart (figure-ground)?",
    "Are the patterns in the data emphasized through the layout or colour choices?"
]

concepts_dict = {"expressiveness": expressiveness_questions, "efficiency": efficiency_questions, "engagement": engagement_questions, "data_ink_ratio": data_ink_ratio_questions, "gestalt_principles": gestalt_principles_questions}