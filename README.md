# Rossmann Store Sales Forecasting

## Project Overview
This project aims to forecast sales for Rossmann Pharmaceuticals across their stores, six weeks in advance. It leverages machine learning and deep learning techniques to predict daily sales, taking into account factors such as promotions, competition, holidays, seasonality, and locality.

## Business Context
Rossmann's finance team requires accurate sales forecasts to facilitate better resource planning and financial decision-making. This project provides an end-to-end solution that delivers these predictions to finance team analysts.

## Project Structure

```plaintext

rossmann-pharmaceuticals-sales-forecast/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml   # GitHub Actions
├── .gitignore              # files and folders to be ignored by git
├── requirements.txt        # contains dependencies for the project
├── README.md               # Documentation for the projects
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
|   |──preprocessing.ipynb              # Jupyter notebook for data cleaning and processing 
|   ├──eda_analysis.ipynb               # Jupyter notebook for customer purchasing behavior analysis 
|   ├──ml_preprocess.ipynb              # Jupyter notebook for  data preparation for model training 
|   ├──ml_modelling.ipynb               # Jupyter notebook for Regression model training 
|   |──dl_modelling.ipynb               # Jupyter notebook for LSTM model training 
│   └── README.md                       # Description of notebooks directory 
├── tests/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    ├── preprocessing.py            # script for data processing, cleaning
    ├── eda_analysis.py             # Script for customer EDA analysis of customer purchasing behavior
    ├── ml_preprocess.py            # script for data processing for machine learning model
    ├── ml_modelling.py             # script for regression model training
    |──  dl_modelling.py            # script for LSTM model training 
    └── README.md                   # Description of scripts directory
    
```

```
## Setup

1. Clone the repository:
   ```
   git clone https://github.com/OL-YAD/rossmann-pharmaceuticals-sales-forecast.git
   cd rossmann-pharmaceuticals-sales-forecast
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Data Source
The dataset used in this project is sourced from the [Rossmann Store Sales|Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data).