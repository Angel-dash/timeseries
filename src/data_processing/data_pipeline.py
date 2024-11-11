# # from ..data.data_fetching import fetch_data_as_dataframe
# import pandas as pd
# import os 
# import sys
# import joblib
# import numpy as np 

# from tensorflow.keras.models import load_model
# from sklearn.metrics import mean_squared_error

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from src.data.data_fetching import DataFetcher
# BASE_URL = 'https://gss.wscada.net/api/socket/mai_test/response'

# # Initialize the class with the base URL and folder path
# data_fetcher = DataFetcher(base_url=BASE_URL)

# # Fetch data and form DataFrame
# df = data_fetcher.fetch_data()

# # Handle missing data and pivot DataFrame
# df_pivoted = data_fetcher.handle_missing_data()

# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print("This is df_pivoted",df_pivoted)
# print("this is df, ", df)
# print("************************************************************************************")

# # Save the DataFrame to a CSV file
# data_fetcher.save_dataframe()

# import joblib

# # Correct path to the scaler
# scaler_path = '/home/nightwing/Codes/LSTM_for_waterlevel_prediction/Notebook/scaler.save'

# # Load the scaler
# scaler = joblib.load(scaler_path)




# # # Usage
# model_path = '/home/nightwing/Codes/LSTM_for_waterlevel_prediction/Notebook/models/best_4_model_v2.h5'



# #Reordering the columns
# def reorder_dataframe(df):
#     df_reorder = df.iloc[:,[0,4,1,5,2,3]]
#     return df_reorder


# #Converting timestamp into standard format 
# def convert_timestamps(df, column_name, target_format='%Y-%m-%d %H:%M:%S'):
#     """
#     Convert timestamps in a specified column of a DataFrame to a target format.
    
#     Parameters:
#     df (pandas.DataFrame): The input DataFrame
#     column_name (str): The name of the column containing timestamps
#     target_format (str): The desired output format (default: '%Y-%m-%d %H:%M:%S')
    
#     Returns:
#     pandas.DataFrame: A new DataFrame with the converted timestamps
#     """
#     # Create a copy of the DataFrame to avoid modifying the original
#     df_copy = df.copy()
    
#     # Convert the specified column to datetime
#     df_copy[column_name] = pd.to_datetime(df_copy[column_name], utc=True)
    
#     # Convert to the target format
#     df_copy[column_name] = df_copy[column_name].dt.strftime(target_format)
    
#     return df_copy

# #Converting datatime as an index

# def set_datetime_index(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
#     """
#     Converts a specified column to datetime format and sets it as the DataFrame index.

#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the datetime column.
#     datetime_column (str): The name of the column to convert and set as the index.

#     Returns:
#     pd.DataFrame: The DataFrame with the datetime column as the index.
#     """
#     # Convert the specified column to datetime format
#     df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

#     # Set the datetime column as the index
#     df.set_index(datetime_column, inplace=True)

#     # Sort the DataFrame by the index to ensure it's in chronological order
#     df.sort_index(inplace=True)

#     return df


# def create_new_dataframe_with_reduced_timestamps(df, hours=1):
#     """
#     Create a new DataFrame by repeatedly subtracting 1 hour from the last timestamp
#     and copying each row's data until the first timestamp is reached.
    
#     Parameters:
#     df (pd.DataFrame): Input DataFrame with the index set to 'Timestamp'.
#     hours (int): The number of hours to subtract for each step (default is 1).
    
#     Returns:
#     pd.DataFrame: A new DataFrame with adjusted timestamps.
#     """
#     # Ensure the index is a DateTimeIndex
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise ValueError("The DataFrame index must be a DateTimeIndex.")
    
#     # Get the last timestamp from the DataFrame
#     last_timestamp = df.index[-1]

#     # List to store the new rows
#     new_rows = []

#     # Loop until we reach the first timestamp
#     while last_timestamp >= df.index[0]:
#         # Get the row corresponding to the current timestamp
#         row_data = df.loc[last_timestamp]

#         # Append the row data to the list
#         new_rows.append(row_data)

#         # Subtract the specified number of hours from the current timestamp
#         last_timestamp -= pd.Timedelta(hours=hours)

#     # Convert the list of rows into a new DataFrame
#     new_df = pd.DataFrame(new_rows)

#     # Reverse the DataFrame to keep the original order
#     new_df = new_df[::-1]

#     # Set the index of the new DataFrame to the timestamps, adjusting to match the loop
#     new_df.index = pd.date_range(end=df.index[-1], periods=len(new_df), freq=f'-{hours}H')

#     return new_df


# def interploate_missing_values_with_time(df: pd.DataFrame, method: str = 'time') -> pd.DataFrame:
#     """
#     Interpolates missing values in a DataFrame using the specified method.
    
#     Parameters:
#     - df (pd.DataFrame): The DataFrame to interpolate.
#     - method (str): The interpolation method to use. Default is 'time'.
    
#     Returns:
#     - pd.DataFrame: The DataFrame with interpolated values.
#     """
#     if method not in ['linear', 'time', 'index', 'nearest', 'polynomial', 'spline']:
#         raise ValueError("Invalid interpolation method. Choose from 'linear', 'time', 'index', 'nearest', 'polynomial', 'spline'.")
    
#     # Print NaN values before interpolation
#     print("NaN values before interpolation:", df.isna().sum())
    
#     # Perform interpolation
#     df_interpolated = df.interpolate(method=method, inplace=False)
    
#     # Print NaN values after interpolation
#     print("NaN values after interpolation:", df_interpolated.isna().sum())
    
#     # Check for any remaining NaN values
#     if df_interpolated.isna().sum().sum() > 0:
#         print("Warning: Some NaN values remain after interpolation.")
#         # Optionally, fill remaining NaNs with a specific method
#         df_interpolated = df_interpolated.fillna(method='ffill').fillna(method='bfill')
#         print("NaN values after additional filling:", df_interpolated.isna().sum())
    
#     return df_interpolated


# def last_48_hours(df):
#     df_last_48_hours = df.tail(48)
#     return df_last_48_hours


# def preprocess_data(df, scaler, n_steps_in):
#     # Ensure the dataframe has the same columns as the training data
#     # Scale the data
#     df_scaled = scaler.transform(df.values)
    
#     # Create sequences
#     X = []
#     for i in range(len(df_scaled) - n_steps_in + 1):
#         X.append(df_scaled[i:i+n_steps_in, :])
    
#     return np.array(X)

# def postprocess_predictions(predictions, scaler):
#     # Convert predictions to a 2D array (n_samples, 1)
#     predictions = np.array(predictions).reshape(-1, 1)
    
#     # Create dummy columns for the rainfall data (which won't be used)
#     dummy_rainfall = np.zeros((predictions.shape[0], 4))  # 4 columns for rainfall

#     # Concatenate the predictions (water levels) with the dummy rainfall data
#     predictions_with_rainfall = np.hstack((dummy_rainfall, predictions))

#     # Apply inverse_transform
#     inverse_scaled_predictions = scaler.inverse_transform(predictions_with_rainfall)

#     # Return only the water level column
#     return inverse_scaled_predictions[:, -1]


# model = load_model(model_path, custom_objects={'mse':mean_squared_error})

# def predict_water_levels(df, n_steps_in, n_steps_out):
#     # Fetch and preprocess data

#     # Ensure you use the same scaler as the one used during training
#     X = preprocess_data(df, scaler, n_steps_in)
#     print("THis is X", X)

#     # Make predictions for the next 4 hours
#     predictions = model.predict(X)

#     print("This is prediction", predictions)

#     # Post-process predictions (inverse transform to get actual water levels)
#     predictions = postprocess_predictions(predictions, scaler)

#     print("This is the real predicted values",predictions)
#     return predictions

# n_steps_in = 48 
# n_steps_out = 4



# import os
# import pandas as pd

# def get_prediction(df):
#     df_reordered = reorder_dataframe(df)
#     df_formatted_timestamp = convert_timestamps(df_reordered, 'Timestamp')
#     df_datetime_index = set_datetime_index(df_formatted_timestamp, 'Timestamp')
#     df_filtered_hourly = create_new_dataframe_with_reduced_timestamps(df_datetime_index)
#     df_interpolated = interploate_missing_values_with_time(df_filtered_hourly, 'time')
#     df_last_48_hours_data = last_48_hours(df_interpolated)
#     print("This is data after interpolation: ")
#     print(df_last_48_hours_data)
#     print(df_last_48_hours_data.isna().sum())  # Add this line to check for NaN values
    
#     # Save df_last_48_hours_data as CSV in test_data folder
#     test_data_folder = 'test_data'
#     if not os.path.exists(test_data_folder):
#         os.makedirs(test_data_folder)
    
#     csv_file_path = os.path.join(test_data_folder, 'last_48_hours_data.csv')
#     df_last_48_hours_data.to_csv(csv_file_path, index=True)
#     print(f"Saved last 48 hours data to: {csv_file_path}")
    
#     # Debug: Print data types and check for infinity values
#     print("Data types:", df_last_48_hours_data.dtypes)
#     print("Infinity check:", np.isinf(df_last_48_hours_data).sum())
    
#     # Preprocess data
#     X = preprocess_data(df_last_48_hours_data, scaler, n_steps_in)
    
#     # After scaling
#     print("Scaled data shape:", X.shape)
#     print("Scaled data sample:", X[0])
#     print("Scaled data stats:", np.nanmin(X), np.nanmax(X), np.nanmean(X))
    
#     # Before prediction
#     print("Model input shape:", X.shape)
#     print("Model input sample:", X[0])
    
#     # Make predictions
#     predictions = model.predict(X)
    
#     # After prediction
#     print("Raw predictions:", predictions)
#     print("Prediction stats:", np.nanmin(predictions), np.nanmax(predictions), np.nanmean(predictions))
    
#     # Post-process predictions
#     predictions = postprocess_predictions(predictions, scaler)
    
#     print("Final predicted values:", predictions)
#     return predictions
# predicted_water_levels = get_prediction(fake_data_df)
# print(predicted_water_levels)