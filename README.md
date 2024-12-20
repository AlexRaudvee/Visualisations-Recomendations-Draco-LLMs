# Combining logic programming with LLMs for visualization recommendation

This project explores the integration of **Draco2**, a logic programming framework for visualization design, with a **Large Language Model (LLM)** for generating and evaluating chart recommendations. The ultimate goal is to recommend expressive and effective visualizations based on the given data, leveraging the strengths of both logic programming and AI.

## Contents

- [Introduction](#introduction)
- [Principles of Visualization](#principles-of-visualization)
  - [Expressiveness](#expressiveness)
  - [Gestalt Principles](#gestalt-principles)
  - [More Principles](#more-principles)
- [Results](#results)
- [How to Run](#how-to-run)
  - [Local Installation](#local-installation)
  - [Using Docker](#using-docker)
- [Repository Structure](#repository-structure)
- [References](#references)

---

## Introduction

This project implements an automated pipeline for visualization recommendation, using the following tools:
- **Draco2**: A constraint solver to recommend visualizations in **Vega-Lite** format.
- **LLM (Gemini API)**: Used to identify relevant columns for visualization from a dataset.
- **Evaluation Metrics**: Combined evaluation using multimodal LLMs and Draco's scoring system, which accounts for soft constraint violations.

The dataset used is the **weather dataset** from Vega-Lite, and the methodology emphasizes key visualization principles like **Expressiveness** and **Gestalt Principles**.

---

## Principles of Visualization

### Expressiveness
A visualization should express all and only the information in the dataset relevant to the task at hand. It ensures the visual encoding matches the data semantics, avoiding misinterpretation or redundancy.

### Gestalt Principles
Derived from Gestalt psychology, these principles help ensure the visualizations leverage human perception. Examples include:
- **Proximity**: Related elements should be visually grouped.
- **Similarity**: Elements sharing similar visual properties (e.g., color, shape) are perceived as related.

### More Principles
*(This section is a placeholder for additional principles like Clarity, Simplicity, and Accuracy.)*

---

## Results

Visualizations generated during the project and their evaluations will be added here.

**[Placeholder for images and links to results.]**

---

## How to Run

This section provides instructions for running the project locally or using Docker.

### Local Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AlexRaudvee/Visualisations-Recomendations-Draco-LLMs
   cd https://github.com/AlexRaudvee/Visualisations-Recomendations-Draco-LLMs
   ```
2. **Set Up a Python Environment**:
    - Create and activate a virtual environment:
    ```bash 
    python3 -m venv venv
    source venv/bin/activate   # For Windows: venv\Scripts\activate
    ```
    - Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Script:
    - Configure the config.json file with your Gemini API key and Draco setup details.
    - Execute the main script:
    ```bash
    python main.py
    ```
    
### Using Docker
1. Build the Docker Image:
```bash
docker build -t visualization-recommender .
```
2. Run the Container:
```bash
docker run -it visualization-recommender
```

## Repository Structure
```plaintext
.
├── data/                  # Input datasets
├── results/               # Generated visualizations and evaluation outputs
├── src/                   # Source code for the project
│   ├── llm_integration.py # LLM-related logic for column selection
│   ├── draco_integration.py # Draco2 logic programming interface
│   ├── evaluation.py      # Evaluation methods using LLMs and Draco scores
│   └── main.py            # Main execution script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # Project documentation
```

## References
- Draco Documentation
- Vega-Lite Documentation
- Gestalt Principles in Data Visualization
- Expressiveness in Visualization