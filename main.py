import pandas as pd
from vega_datasets import data

# Loading data
weather_data: pd.DataFrame = data.seattle_weather()
# Separating out the positive temperatures for the log scale example
weather_data_positive = weather_data[weather_data["temp_max"] > 0]

import json

from IPython.display import display

from draco import Draco, dict_to_facts, dict_union
from draco.renderer import AltairRenderer

d = Draco()
renderer = AltairRenderer(concat_mode="hconcat")


def show(*args, df: pd.DataFrame = weather_data):
    spec = dict_union(*args)
    prog = dict_to_facts(spec)
    if not d.check_spec(prog):
        print("\n".join(prog))
        print(d.get_violations(prog))
        assert False, "Invalid spec"

    # Display the rendered VL chart and the ASP
    chart = renderer.render(spec, df)
    print(json.dumps(spec, indent=2))
    display(chart)
    display(prog)
    
    
def data(fields: list[str], df: pd.DataFrame = weather_data) -> dict:
    number_rows, _ = df.shape
    return {
        "number_rows": number_rows,
        "field": [
            x
            for x in [
                {"name": "temp_max", "type": "number", "__id__": "temp_max"},
                {"name": "wind", "type": "number", "__id__": "wind"},
                {"name": "precipitation", "type": "number", "__id__": "precipitation"},
                {"name": "weather", "type": "string", "__id__": "weather"},
            ]
            if x["name"] in fields
        ],
    }
    
show(
    data(["weather", "temp_max"]),
    {
        "view": [
            {
                "coordinates": "cartesian",
                "mark": [
                    {
                        "type": "bar",
                        "encoding": [
                            {"channel": "x", "field": "weather"},
                            {
                                "channel": "y",
                                "field": "temp_max",
                                "aggregate": "mean",
                            },
                        ],
                    }
                ],
                "scale": [
                    {"channel": "x", "type": "ordinal"},
                    {"channel": "y", "type": "linear", "zero": "true"},
                ],
            }
        ]
    },
)