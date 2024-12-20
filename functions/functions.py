#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# Imports for working with json and strings 
import os
import csv
import ast
import json
import time
import itertools
import PIL.Image

# Display utilities and math
import numpy as np
from IPython.display import Markdown, display

import draco as drc
import pandas as pd                # pandas to manipulate the data
import altair as alt               # Altair to manipulate the knowledge and the graphs
import vl_convert as vlc

from draco.renderer import AltairRenderer
from config import *

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


def generate_summary(df: pd.DataFrame):
    """
    Generate a concise summary of a DataFrame's columns, non-null counts, and data types.

    Description
    ---
    This function produces a readable summary of the input pandas DataFrame. The summary includes:
      - Index of each column.
      - Column names.
      - Non-null count for each column.
      - Data type of each column.
    It also provides an aggregated count of unique data types in the DataFrame.

    Parameters
    ---
    - df : (pd.DataFrame)
        The input pandas DataFrame for which the summary is to be generated.

    Returns
    ---
    - str : A formatted string containing the column index, column name, non-null counts, data types, 
            and a summary of all data types with their respective counts.

    Raises
    ---
    - AttributeError: If the input `df` is not a pandas DataFrame or does not have valid attributes like `columns` or `dtypes`.

    Examples
    ---
    >>> import pandas as pd
    >>> data = {
    ...     "col1": [1, 2, None, 4],
    ...     "col2": ["a", "b", "c", None],
    ...     "col3": [1.1, 2.2, 3.3, 4.4]
    ... }
    >>> df = pd.DataFrame(data)
    >>> print(generate_summary(df))
    ---   Column         Non-Null Count  Dtype
    ---  ------         --------------  -----
    0    col1           3               float64
    1    col2           3               object
    2    col3           4               float64
    dtypes: float64(2), object(1)
    """
    summary = []
    summary.append("---   Column         Non-Null Count  Dtype")
    summary.append("---  ------         --------------  -----")
    for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
        non_null_count = df[col].notnull().sum()
        summary.append(f"{i:<4} {col:<15} {non_null_count:<15} {dtype}")
    dtype_summary = ", ".join([f"{dtype.name}({(df.dtypes == dtype).sum()})" for dtype in df.dtypes.unique()])
    summary.append(f"dtypes: {dtype_summary}")
    return "\n".join(summary)


def extract_python_list(input_string: str):
    """
    Extract a Python list from a string containing Python code.

    Description
    ---
    This function takes an input string that represents a Python list, potentially wrapped 
    in markdown-style code block delimiters (e.g., ```python ... ```). It removes these 
    delimiters and safely evaluates the string into an actual Python list using `ast.literal_eval`. 
    This ensures the evaluation is safe and avoids security risks posed by `eval()`.

    Parameters
    ---
    - input_string : (str)
        A string containing a Python list, optionally wrapped in markdown-style code 
        block delimiters such as ```python and ```.

    Returns
    ---
    - list : 
        A Python list object extracted from the input string.

    Raises
    ---
    - ValueError: If the string does not contain a valid Python list.
    - SyntaxError: If the input string is not properly formatted as Python code.
    - TypeError: If the input is not a string.

    Examples
    ---
    >>> input_str = \"\"\"```python
    ... [1, 2, 3, 4]
    ... ```\"\"\"
    >>> extract_python_list(input_str)
    [1, 2, 3, 4]

    >>> input_str = "[10, 20, 30]"
    >>> extract_python_list(input_str)
    [10, 20, 30]

    >>> input_str = \"\"\"```python
    ... ['apple', 'banana', 'cherry']
    ... ```\"\"\"
    >>> extract_python_list(input_str)
    ['apple', 'banana', 'cherry']

    Notes
    ---
    - This function uses `ast.literal_eval` for evaluation, which only allows Python literals 
      (e.g., lists, dictionaries, strings, integers, etc.) and is safer than `eval()`.
    - Ensure the input string contains valid Python code that represents a list.
    """
    # Remove the markdown-style code block delimiters
    cleaned_string = input_string.replace("```python", "").replace("```", "").strip()
    # Use ast.literal_eval to safely evaluate the Python list
    python_list = ast.literal_eval(cleaned_string)
    return python_list


def recommend_columns(model, df: pd.DataFrame):
    """
    Recommend two columns from a DataFrame for the most expressive visualization.

    Description
    ---
    This function generates a prompt containing a summary of the input DataFrame 
    (created using the `generate_summary` function) and asks the provided model 
    (e.g., a language model) to recommend two columns that would result in the 
    most expressive visualization. The model's response is parsed into a Python 
    list using the `extract_python_list` function, and the recommended column 
    names are returned.

    Parameters
    ---
    - model : 
        A model object capable of generating content based on a text prompt. The 
        model must have a `generate_content` method that accepts a string prompt 
        and returns a response with a `.text` attribute.

    - df : (pd.DataFrame)
        The input pandas DataFrame whose columns are to be analyzed for the 
        recommendation.

    Returns
    ---
    - list : 
        A list containing exactly two column names as strings, recommended by the model.

    Raises
    ---
    - AssertionError: If the model's response is not a list.
    - ValueError: If the response does not contain exactly two elements.
    - AttributeError: If the `model` object does not have the expected `generate_content` method or response format.

    Examples
    ---
    >>> import pandas as pd
    >>> class MockModel:
    ...     def generate_content(self, contents):
    ...         class Response:
    ...             text = "['col1', 'col2']"
    ...         return Response()
    ...
    >>> data = {
    ...     "col1": [1, 2, 3],
    ...     "col2": [4, 5, 6],
    ...     "col3": [7, 8, 9]
    ... }
    >>> df = pd.DataFrame(data)
    >>> model = MockModel()
    >>> recommend_columns(model, df)
    ['col1', 'col2']

    Notes
    ---
    - The `generate_summary` function is used to produce a summary of the DataFrame 
      to provide context for the model.
    - The `extract_python_list` function ensures that the model's response is safely 
      converted into a Python list.
    - The function asserts that the returned object is a list and assumes that the 
      model provides a valid response with exactly two column names.
    """
    prompt = f"""Given this information about the dataframe:\n\n{generate_summary(df)}\n\n
                 Give me the list of two columns that are going to result in the 
                 most expressive visualisation, your answer should contain only 
                 this python list [column, column].
            """
    
    _response = model.generate_content(contents=prompt)
    response = _response.text
    response = extract_python_list(response)

    assert type(response) == list

    return response 


def generate_column_combinations(df: pd.DataFrame):
    """
    Generate all unique combinations of two columns from a DataFrame.

    Description
    ---
    This function takes a pandas DataFrame as input and generates all possible unique 
    combinations of two columns. The resulting combinations are returned as a list of lists, 
    where each sublist contains the names of two columns.

    Parameters
    ---
    - df : (pd.DataFrame)
        The input pandas DataFrame whose column combinations are to be generated.

    Returns
    ---
    - list : 
        A list of lists, where each sublist contains two column names (as strings) 
        representing a unique combination.

    Raises
    ---
    - AttributeError: If the input `df` does not have a `columns` attribute (i.e., it is not a valid DataFrame).
    - ValueError: If the DataFrame has fewer than two columns, as no combinations can be generated.

    Examples
    ---
    >>> import pandas as pd
    >>> data = {
    ...     "col1": [1, 2, 3],
    ...     "col2": [4, 5, 6],
    ...     "col3": [7, 8, 9]
    ... }
    >>> df = pd.DataFrame(data)
    >>> generate_column_combinations(df)
    [['col1', 'col2'], ['col1', 'col3'], ['col2', 'col3']]

    >>> data = {"col1": [1, 2], "col2": [3, 4]}
    >>> df = pd.DataFrame(data)
    >>> generate_column_combinations(df)
    [['col1', 'col2']]

    Notes
    ---
    - The function uses `itertools.combinations` to generate unique combinations of two columns.
    - If the DataFrame has fewer than two columns, no valid combinations can be produced.
    - The output is formatted as a list of lists for ease of further processing.
    """

    # Get the list of columns from the dataframe
    columns = df.columns.tolist()
    
    # Generate all unique combinations of two columns
    combinations = list(itertools.combinations(columns, 2))
    
    # Convert each combination to a list format
    combinations = [list(comb) for comb in combinations]
    
    return combinations


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
    - tuple:
        A tuple of the top design
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
    
    first_key = next(iter(recommendation_dict))
    top_design = recommendation_dict[first_key]

    return sorted_dict, top_design


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
    # display(chart)

    file_name += ".png"

    # Save the chart to a png file
    png_data = vlc.vegalite_to_png(vl_spec=chart.to_dict())
    with open(file_name, "wb") as f:
        f.write(png_data)


def evaluate_chart_with_LLM(path_to_chart: str, question_set: list[str]) -> list[float] | None:
    """
    Evaluate a chart using a large language model (LLM) based on a set of questions.

    Description
    ---
    This function evaluates a chart image (specified by its file path) and a set of questions 
    related to the chart by passing both to an evaluation model (e.g., a large language model). 
    The model generates a response which is then cleaned, parsed, and returned as a list of floats 
    corresponding to the answers to the questions. If an error occurs during the evaluation process, 
    the function returns `None`.

    Parameters
    ---
    - path_to_chart : (str)
        The file path to the chart image that will be evaluated by the model.

    - question_set : (list[str])
        A list of questions related to the chart, which will be used to prompt the model.

    Returns
    ---
    - list[float] : 
        A list of float values representing the model's responses to each question in the set.
        Each float corresponds to an answer from the model for a given question.
    - None : 
        If any exception occurs during the process (e.g., file loading issues, model errors, etc.), the function returns `None`.

    Raises
    ---
    - FileNotFoundError: If the chart image file cannot be found at the given path.
    - ValueError: If the model's response is not in the expected format (cannot be parsed as a list of floats).
    - TypeError: If the `question_set` is not a list of strings or if the returned response cannot be processed.

    Examples
    ---
    >>> question_set = ["What is the trend of the data?", "What is the maximum value?"]
    >>> evaluate_chart_with_LLM("path/to/chart.png", question_set)
    [0.85, 12.34]  # Example output representing the model's responses

    >>> evaluate_chart_with_LLM("invalid/path/to/chart.png", question_set)
    None  # If the file path is incorrect or any exception occurs

    Notes
    ---
    - This function relies on the `evaluation_model.generate_content()` method to generate answers.
    - The returned response is expected to be a string that can be parsed into a Python list.
    - If the model's response is not in the expected format (a list of floats), the function will raise an error and return `None`.
    - The function uses the PIL library to open the chart image. Ensure that PIL (Pillow) is installed and the image is accessible.

    Dependencies
    ---
    - PIL (Pillow): Used to open the image file.
    - ast: Used to safely evaluate and parse the model's response into a list.
    - evaluation_model: A model that can generate content based on a prompt and image input.
    """
    
    try:
        sample_file = PIL.Image.open(f"{path_to_chart}")
        prompt = f"QUESTIONS:{question_set}"
        response_ = evaluation_model.generate_content([prompt, sample_file]).text

        # Clean and parse the string
        cleaned_string = response_.strip("```python\n").strip("\n```")
        response = ast.literal_eval(cleaned_string)
        return response

    except Exception as e:
        # print(e)
        return None
    
    
def apply_eval_to_charts_folder(func, directory: str, output_csv: str, concepts_dict: dict):
    """
    Apply a specified evaluation function to all files in a directory, process the results, 
    and save them to a CSV file.

    Description
    ---
    This function walks through all files in the specified directory, applies a user-defined 
    evaluation function (`func`) to each file, and processes the results for each concept defined 
    in the `concepts_dict`. For each file, if the evaluation function produces a result, it appends 
    the results in a list. Once all files are processed, the results are written to a CSV file.

    Parameters
    ---
    - directory : (str)
        The directory path containing the files to be processed. The function will recursively 
        iterate through all files in this directory.

    - output_csv : (str)
        The path where the results will be saved as a CSV file. This CSV will contain columns 
        for file information, concepts, and evaluation results.

    - func : (function)
        A user-defined function to evaluate each file. The function must accept two arguments: 
        the file path and a concept (string) from the `concepts_dict`, and return a result (any type).

    - concepts_dict : (dict)
        A dictionary where the keys are concept names and the values are the associated data 
        that should be passed to `func` for evaluation.

    Returns
    ---
    - None
        This function does not return any value. It writes the results directly to the specified CSV file.

    Raises
    ---
    - ValueError: If any of the files in the directory cannot be processed or if the function 
      encounters an issue with the concept data.
    - FileNotFoundError: If the specified directory or output CSV path does not exist.
    - TypeError: If the `func` is not callable or `concepts_dict` is not a dictionary.
    - Exception: If the `func` fails or returns an unexpected result that cannot be processed.

    Examples
    ---
    >>> import os
    >>> def mock_func(file_path, concept):
    >>>     return f"Result for {concept} in {file_path}"
    >>>
    >>> concepts = {'concept1': 'data1', 'concept2': 'data2'}
    >>> apply_eval_to_charts_folder("path/to/charts", "output.csv", mock_func, concepts)

    Notes
    ---
    - The function performs a retry mechanism when the `func` does not return a valid result, 
      waiting for 20 seconds before retrying.
    - The file names are expected to contain information that is split by "+" and used in the result.
    - The function writes a CSV file with headers `['col1', 'col2', 'draco_score', 'concept', 'Result']`.
    """
    
    # Prepare the results list
    results = []

    # Iterate through files in the directory
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            for key in concepts_dict:
                retry = True
                while retry:
                    # Apply the function to the file and store the result
                    result = func(file_path, concepts_dict[f'{key}'])
                    if result:
                        results.append((file_name[:-4].split("+")[0],
                                        file_name[:-4].split("+")[1],
                                        file_name[:-4].split("+")[2], key, result))
                        retry = False
                        print(f"DONE: {file_name}, {key}")
                    else:
                        # Handle errors and record them in the results
                        print(f"ERROR: Error in LLM output for concept '{key}' for chart {file_name[:-4]}. Cool-down of LLM for 20 seconds...")
                        time.sleep(20)
                    

    # Write results to a CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['col1', 'col2', 'draco_score', 'concept', 'Result'])  # Header row
        writer.writerows(results)

    print(f"Results have been saved to {output_csv}")