# Flood Forecasting and Weather Prediction

This repository contains the data pipeline, experiments, and models used for flood forecasting and weather prediction using time-series data.

## Installation

To install all the required libraries, run the following command:

```bash
pip install -r requirements.txt

## Project Structure
/Data

    socket_client.py
    Contains the pipeline for fetching and preprocessing datasets using sockets in Python. The resulting processed dataset will be saved inside the /Data/processed directory.

/Dataset

    Contains CSV files of different stations used to train the model. It includes:
        Hourly_rainfall.csv: Data on hourly rainfall.
        Hourly_waterfall.csv: Data on hourly waterfall measurements.

These datasets are used for predictions and testing the model.
/Notebooks

    Contains .ipynb files used for experiments:
        csv_datapipeline.ipynb
        This notebook is used for loading the model, performing all the data preprocessing steps, and pushing the resulting predictions to forecast.wscada. Running every cell in the notebook is enough to push the processed data.
        flood_forecasting.ipynb
        This notebook is used to train the best model, which takes 48 hours of data and predicts the next 4 hours. The model trained in this notebook is saved as best_4_model_v2.h5.
        time_series_weather_forecasting.ipynb
        This is where the initial experiments were conducted. The best model trained on 24-hour data is saved as best_model.h5.

/Notebooks/models

    Contains all the models trained during the experiments, including:
        best_model.h5: The model trained on the last 24 hours of weather data.
        best_4_model_v2.h5: The model trained on 48 hours of data to predict the next 4 hours.

Usage

    Fetch and preprocess data using the socket_client.py script.
    Use the Jupyter notebooks in the /Notebooks directory for running experiments, preprocessing data, training models, and pushing predictions.
    Trained models are saved in /Notebooks/models and can be used for further predictions or testing.
