import pandas as pd 
import numpy as np 
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import sys
import os
import joblib
import json 

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data_processing.data_preprocessing_pipeline import set_datetime_index,create_sequences
from src.data_processing.data_preprocessing_pipeline import interploate_missing_values_with_time, reorder_dataframe
from src.data_processing.data_postprocessing import generate_prediction_timestamps,create_dateframe_with_timestamps,convert_predictions_to_json
from src.data_processing.post_data import send_predictions

df_rainfall = pd.read_csv("/home/nightwing/Codes/LSTM_for_waterlevel_prediction/Dataset/Hourly_rainfall_data.csv")
df_waterlevel = pd.read_csv("/home/nightwing/Codes/LSTM_for_waterlevel_prediction/Dataset/Waterlevel_hourly.csv")

url = "http://forecast.wscada.net/import"
origin_code = "500239"
parameter_code = "WL_1H"

model_path = '/home/nightwing/Codes/LSTM_for_waterlevel_prediction/Notebook/models/best_4_model_v2.h5'
model = load_model(model_path, custom_objects={'mse':mean_squared_error})

# Correct path to the scaler
scaler_path = '/home/nightwing/Codes/LSTM_for_waterlevel_prediction/Notebook/scaler.save'

# Load the scaler
scaler = joblib.load(scaler_path)

features = ['Mai Beni R (mm) (Sum)', 'Nayabazar - Namsaling R (mm) (Sum)', 
            'Pashupatingar R (mm) (Sum)', 'Sandakpur - Valley R (mm) (Sum)']
target = 'Mai Khola WTR_LVL (m) (Avg)'

def data_preprocessing(df_rainfall, df_waterlevel, scaler):
    df_datetime_index_rainfall = set_datetime_index(df_rainfall,'Time')
    df_datetime_index_waterlevel = set_datetime_index(df_waterlevel,'Time')
    combined_df = pd.merge(df_datetime_index_rainfall, df_datetime_index_waterlevel, left_index=True, right_index=True, how='outer')
    df_interploated_missing_values = interploate_missing_values_with_time(combined_df)
    df_reordered = reorder_dataframe(df_interploated_missing_values)
    df_reordered_scaled = pd.DataFrame(scaler.fit_transform(df_reordered), columns=df_reordered.columns, index=df_reordered.index)
    return df_reordered_scaled

df_reordered_scaled = data_preprocessing(df_rainfall, df_waterlevel, scaler)

X, y = create_sequences(df_reordered_scaled.values, n_past=48, n_future=4)

def get_prediction(X):
    predictions_scaled = model.predict(X)
    predictions_reshaped = predictions_scaled.reshape(-1,1)
    dummy_features = np.zeros((predictions_reshaped.shape[0], len(features)))
    inverse_ready = np.hstack((dummy_features, predictions_reshaped))
    predictions = scaler.inverse_transform(inverse_ready)[:, -1]
    return predictions

def get_prediction_with_timestamps(X):
    predicted_values = get_prediction(X)

    # prediction_timestamps = generate_prediction_timestamps(X,df_reordered_scaled)

    # all_predictions_df = create_predictions_dataframe(predicted_values,prediction_timestamps)

    all_predictions_df = create_dateframe_with_timestamps(predicted_values,df_reordered_scaled)

    return all_predictions_df


all_predictions_df = get_prediction_with_timestamps(X)

json_data_prediction = convert_predictions_to_json(all_predictions_df)
# print(json.dumps(json_data_prediction, indent=2))

response = send_predictions(all_predictions_df, url, origin_code, parameter_code)