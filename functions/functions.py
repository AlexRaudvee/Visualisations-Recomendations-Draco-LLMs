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
from altair_saver import save

import draco as drc

# Initialize draco and Altair Render
d = drc.Draco()
renderer = AltairRenderer()

# Handles serialization of common numpy datatypes
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def md(markdown: str):
    display(Markdown(markdown))


def pprint(obj):
    md(f"```json\n{json.dumps(obj, indent=2, cls=NpEncoder)}\n```")


def recommend_charts(spec: list[str], df: pd.DataFrame, num: int = 5, labeler=lambda i: f"CHART {i+1}") -> dict[str, tuple[list[str], dict]]:
    """
    Generate and recommend visualizations for a given dataset based on provided chart specifications.

    Description
    ---
    This function uses a recommendation engine to generate and rank visualizations based on
    the provided specifications. It returns a dictionary of chart recommendations, including
    associated facts, specifications, cost metrics, and rendered charts. The charts can be customized 
    (e.g., column-faceted layouts) and ranked based on cost.

    Parameters
    ---
    - spec : (list[str])
        A list of strings representing chart specifications or constraints.
    - df : (pd.DataFrame)
        The dataset to be used for generating visualizations.
    - num : (int, optional)
        The number of charts to generate. Defaults to 5.
    - labeler : (callable, optional)
        A function to customize chart labels. Defaults to labeling charts as "CHART 1", "CHART 2", etc.

    Returns
    ---
    - dict[str, tuple[list[str], dict, float, alt.Chart]]:
        A dictionary where keys are chart names (e.g., "CHART 1") and values are tuples containing:
        - A list of facts derived from the chart's specification.
        - The chart's specification as a dictionary.
        - The cost of the chart based on the recommendation model.
        - The rendered chart object.

    Examples
    ---
    >>> specs = ["bar chart with x=age and y=salary"]
    >>> df = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 70000]})
    >>> chart_recommendations = recommend_charts(specs, df, num=3)
    >>> chart_recommendations["CHART 1"]  # Access the first recommended chart's details.
    """
    
    # Dictionary to store the generated recommendations, keyed by chart name
    chart_specs = {}
    for i, model in enumerate(d.complete_spec(spec, num)):
        chart_name = labeler(i)
        spec = drc.answer_set_to_dict(model.answer_set)

        chart = renderer.render(spec=spec, data=df)
        # Adjust column-faceted chart size if needed
        if (
            isinstance(chart, alt.FacetChart)
            and chart.facet.column is not alt.Undefined
        ):
            chart = chart.configure_view(continuousWidth=130, continuousHeight=130)

        cost = model.cost[0]
        facts = drc.dict_to_facts(spec)

        chart_specs[chart_name] = facts, spec, cost, chart

    return chart_specs


def take_top(recommendation_dict: dict) -> dict:
    """
    Sort chart recommendations by cost in ascending order.

    Description
    ---
    This function reorders a dictionary of chart recommendations based on the cost metric
    (stored as the pre-last element of the tuple associated with each chart). The function
    returns a new dictionary with chart recommendations sorted from lowest to highest cost.

    Parameters
    ---
    - recommendation_dict : (dict) 
        A dictionary of chart recommendations where:
        - Keys are chart names (e.g., "CHART 1").
        - Values are tuples containing facts, specifications, cost metrics, and chart objects.

    Returns
    ---
    - dict: 
        A sorted dictionary where chart recommendations are ordered by their cost in ascending order.

    Examples
    ---
    >>> recommendations = {
    ...     "CHART 1": (["fact1"], {"spec1": "value"}, 10, chart1),
    ...     "CHART 2": (["fact2"], {"spec2": "value"}, 5, chart2),
    ... }
    >>> sorted_recommendations = take_top(recommendations)
    >>> list(sorted_recommendations.keys())  # Output: ["CHART 2", "CHART 1"]
    """

    # Sort the dictionary items based on the cost (pre-last element of the tuple)
    sorted_items = sorted(recommendation_dict.items(), key=lambda item: item[1][-2])
    
    # Convert the sorted items back into a dictionary
    sorted_dict = dict(sorted_items)
    
    return sorted_dict


def display_chart(recommendation_dict: dict, file_name: str = f'assets/'):
    """
    Display the best chart visualization from a dictionary of recommendations.

    Description
    ---
    This function selects and displays the best chart visualization from a recommendation dictionary. 
    The "best" chart is assumed to be the first key-value pair in the dictionary, 
    where the dictionary is typically sorted based on some criteria (e.g., cost or relevance).
    The displayed chart is the last element in the tuple associated with the selected key.

    Parameters
    ---
    - recommendation_dict : (dict) 
        A dictionary where keys are chart names and values are tuples containing:
        - A list of facts associated with the chart.
        - A dictionary representing the chart's specification.
        - A numerical cost metric for the chart.
        - The chart object to be displayed.
    - file_name : (str) - optional
        Path where to store the visualization 
    Returns
    ---
    None : This function directly renders the chart visualization.

    Examples
    ---
    >>> chart_specs = {
    ...     "CHART 1": (["fact1"], {"spec1": "value"}, 10, chart1),
    ...     "CHART 2": (["fact2"], {"spec2": "value"}, 5, chart2),
    ... }
    >>> display_chart(chart_specs)
    (Displays the chart associated with "CHART 1" if it is the first in the sorted dictionary)
    """
    # Take the first element as it is the best visualization
    first_key = next(iter(recommendation_dict))
    chart = recommendation_dict[first_key][-1]

    # Display the chart
    display(chart)

    file_name += f"fig-{first_key}.html"

    # Save the chart to a file
    chart.save(file_name)
