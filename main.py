#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import draco as drc
import pandas as pd

from config import model
from functions import *
from vega_datasets import data as vega_data


# Loading data to be explored
df: pd.DataFrame = vega_data.seattle_weather()
df.head()

# At the beginning we extract the schema of the data that we have
data_schema = drc.schema_from_dataframe(df)
# pprint(data_schema)

# Now we can convert this schema to the facts that Draco can use later to reason about the data when generating recommendations
data_schema_facts = drc.dict_to_facts(data_schema)
# pprint(data_schema_facts)

# Partial specification 
input_spec_base = data_schema_facts + [
    "entity(view,root,v0).",
    "entity(mark,v0,m0).",
]

# Choose the columns with LLM
recommended_columns = recommend_columns(model=model, df=df)

# Extended specification 
input_spec = input_spec_base + [
    # We want to encode the `date` field
    "entity(encoding,m0,e0).",
    f"attribute((encoding,field),e0,{recommended_columns[0]}).",
    # We want to encode the `temp_max` field
    "entity(encoding,m0,e1).",
    f"attribute((encoding,field),e1,{recommended_columns[1]}).",
]

# Make Recommendations
recommendations = recommend_charts(spec=input_spec, df=df, num=5)

# Take the top among Design Space
recommendation, top_viz = take_top(recommendation_dict=recommendations)

# Evaluation Part 
draco_score = top_viz[2]

# Display the best chart among space and save it
chart = top_viz[3]
display_chart(recommendation_dict=recommendations, file_name="assets/best_chart_llm")

print(f"Draco score of the best char by using LLM: {draco_score}")

column_combinations = generate_column_combinations(df=df)
for column_combination in column_combinations:
    pass