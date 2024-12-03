#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

# Display utilities
import json

import numpy as np
from IPython.display import Markdown, display

import altair as alt
import pandas as pd
from vega_datasets import data as vega_data
from draco.renderer import AltairRenderer

import draco as drc

from functions import *

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

########## LATER CHOOSE THE COLUMNS VIA LLM ##########

# Extended specification 
input_spec = input_spec_base + [
    # We want to encode the `date` field
    "entity(encoding,m0,e0).",
    "attribute((encoding,field),e0,date).",
    # We want to encode the `temp_max` field
    "entity(encoding,m0,e1).",
    "attribute((encoding,field),e1,temp_max).",
]

# Make Recommendations
recommendations = recommend_charts(spec=input_spec, df=df, num=5)

# Take the top among Design Space
recommendations = take_top(recommendation_dict=recommendations)

# Display the best chart among space
display_chart(recommendation_dict=recommendations)
