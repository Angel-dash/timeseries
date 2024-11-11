import pandas as pd

def generate_prediction_timestamps(X, df_reordered_scaled, lookback_period=48, prediction_periods=4, freq='H'):
    """
    Generates a list of prediction timestamps based on the input data.

    Parameters:
    X (array-like): Input data, used to determine the number of predictions.
    df_reordered_scaled (pd.DataFrame): The DataFrame containing the timestamps.
    lookback_period (int): The number of previous time steps to consider before making predictions (default: 48).
    prediction_periods (int): The number of prediction periods (default: 4).
    freq (str): The frequency for generating prediction timestamps (default: 'H' for hourly).
    
    Returns:
    list: A list of prediction timestamps.
    """
    prediction_timestamps = []

    for i in range(len(X)):
        # Start time for the current prediction block
        start_time = df_reordered_scaled.index[i + lookback_period]
        
        # Generate timestamps for the next prediction_periods (e.g., 4 hours)
        timestamps = pd.date_range(start=start_time, periods=prediction_periods, freq=freq)
        
        # Extend the list with the generated timestamps
        prediction_timestamps.extend(timestamps)
    
    return prediction_timestamps

def create_predictions_dataframe(predictions, prediction_timestamps, column_name='Predicted Water Level'):
    """
    Creates a DataFrame with predictions and their corresponding timestamps, and sorts it by index.

    Parameters:
    predictions (array-like): The predicted values to be placed in the DataFrame.
    prediction_timestamps (array-like): The timestamps corresponding to each prediction.
    column_name (str): The name of the column for the predicted values (default: 'Predicted Water Level').

    Returns:
    pd.DataFrame: A DataFrame with predictions and sorted by timestamps.
    """
    # Create a DataFrame using predictions and timestamps
    all_predictions_df = pd.DataFrame({column_name: predictions}, index=prediction_timestamps)

    # Sort the DataFrame by its index (timestamps) to ensure chronological order
    all_predictions_df = all_predictions_df.sort_index()

    return all_predictions_df



def create_dateframe_with_timestamps(predictions, df_reordered_scaled, start_date_index=48, end_date_index = -3):

    # Reshape predictions back to (n_samples, 4)
    predictions = predictions.reshape(-1, 4)

    # Generate timestamps for all predictions
    prediction_timestamps = df_reordered_scaled.index[start_date_index:end_date_index]  # Start from the 49th row, end 3 rows before the last
    predictions = predictions[:len(prediction_timestamps)]
    all_predictions_df = pd.DataFrame(
        predictions, 
        index=prediction_timestamps,
        columns=['t+1 prediction', 't+2 prediction', 't+3 prediction', 't+4 prediction']
    )
    return all_predictions_df

def convert_predictions_to_json(all_predictions_df, origin_code="500239", parameter_code="WL_1H"):
    """
    Convert the t+1 predictions from the dataframe to a list of JSON objects with a specific format.
    
    Parameters:
        all_predictions_df (pd.DataFrame): DataFrame containing predictions.
        origin_code (str): Placeholder for the origin code.
        parameter_code (str): Placeholder for the parameter code.
    
    Returns:
        list: A list of dictionaries in JSON format.
    """
    json_list = []
    
    for index, row in all_predictions_df.iterrows():
        json_obj = {
            "origin_code": origin_code,
            "parameter_code": parameter_code,
            "time": index.strftime("%Y-%m-%dT%H:%M:%S"),  
            "value": row['t+1 prediction']  
        }
        json_list.append(json_obj)
    
    return json_list