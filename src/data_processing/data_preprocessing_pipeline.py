import pandas as pd
import numpy as np 

def set_datetime_index(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Converts a specified column to datetime format and sets it as the DataFrame index.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the datetime column.
    datetime_column (str): The name of the column to convert and set as the index.

    Returns:
    pd.DataFrame: The DataFrame with the datetime column as the index.
    """
    # Convert the specified column to datetime format
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

    # Set the datetime column as the index
    df.set_index(datetime_column, inplace=True)

    # Sort the DataFrame by the index to ensure it's in chronological order
    df.sort_index(inplace=True)

    return df

def interploate_missing_values_with_time(df: pd.DataFrame, method: str = 'time') -> pd.DataFrame:
    """
    Interpolates missing values in a DataFrame using the specified method.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to interpolate.
    - method (str): The interpolation method to use. Default is 'time'.
    
    Returns:
    - pd.DataFrame: The DataFrame with interpolated values.
    """
    if method not in ['linear', 'time', 'index', 'nearest', 'polynomial', 'spline']:
        raise ValueError("Invalid interpolation method. Choose from 'linear', 'time', 'index', 'nearest', 'polynomial', 'spline'.")
    
    # Print NaN values before interpolation
    print("NaN values before interpolation:", df.isna().sum())
    
    # Perform interpolation
    df_interpolated = df.interpolate(method=method, inplace=False)
    
    # Print NaN values after interpolation
    print("NaN values after interpolation:", df_interpolated.isna().sum())
    
    # Check for any remaining NaN values
    if df_interpolated.isna().sum().sum() > 0:
        print("Warning: Some NaN values remain after interpolation.")
        # Optionally, fill remaining NaNs with a specific method
        df_interpolated = df_interpolated.fillna(method='ffill').fillna(method='bfill')
        print("NaN values after additional filling:", df_interpolated.isna().sum())
    
    return df_interpolated

def reorder_dataframe(df):
    df_reorder = df.iloc[:,[0,1,2,4,5,]]
    return df_reorder

import joblib

# Correct path to the scaler
scaler_path = '/home/nightwing/Codes/LSTM_for_waterlevel_prediction/Notebook/scaler.save'

# Load the scaler
scaler = joblib.load(scaler_path)
features = ['Mai Beni R (mm) (Sum)', 'Nayabazar - Namsaling R (mm) (Sum)', 
            'Pashupatingar R (mm) (Sum)', 'Sandakpur - Valley R (mm) (Sum)']
target = 'Mai Khola WTR_LVL (m) (Avg)'
# combined_df = pd.merge(df_datetime_index_rainfall, df_datetime_index_waterlevel, left_index=True, right_index=True, how='outer')
# df_reordered_scaled = pd.DataFrame(scaler.fit_transform(df_reordered), columns=df_reordered.columns, index=df_reordered.index)

def create_sequences(data, n_past=48, n_future=4):
    X, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        X.append(data[i:(i + n_past)])
        y.append(data[(i + n_past):(i + n_past + n_future), -1])
    return np.array(X), np.array(y)


