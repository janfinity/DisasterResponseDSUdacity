
# Disaster Response Pipeline Project
## Introduction
As part of the Udacity Data Scientist Nano Degree the objective of this project was to build both an ETL and a ML-Pipeline.
The data is provided by Figure Eight and contains pre-labelled tweets and messages of real-life disaster events. In the course of the projects messages are processed using a natural language processing (NLP) model to categorize messages in real-time.

Therefore the project is structured into three parts:
- Process the data building an ETL-Pipeline
- Building a ML-Pipeline
- Display the results and provide a UI in a web app

## Dependencies
    - Python 3.5+
    - ML-Libraries: NumPy, SciPy, Pandas, scikit-learn
    - NLP-Libraries: NLTK
    - SQLite Libraries: SQLAlchemy
    - Model Dumb and Load: Pickle
    - Web-App: Flask
    - Visualization: Plotly

## Installation

You can find all files in a git repository

git clone https://github.com/

## Execution

- To run the ETL-Pipeline run the following command
    
    'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db'

- To run the ML-Pipeline run the following command

    'python models/train_classifier.py disaster_response.db classifier.pkl'

- To start the web-app run

    'python app/run.py'

