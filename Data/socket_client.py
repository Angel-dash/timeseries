import os
import pandas as pd
from datetime import datetime
import requests

SAVE_FOLDER = os.path.expanduser('~/Codes/LSTM_for_waterlevel_prediction/Data/processed')
BASE_URL = 'https://gss.wscada.net/api/socket/mai_test/response'
response = requests.get(BASE_URL)

if response.status_code == 200:
    data = response.json()
    all_data = []
    
    # List to keep track of stations and parameters for which no data was found
    missing_data = []

    for station in data:
        station_name = station['name']
        observations = station['observations']

        for observation in observations:
            parameter_name = observation['parameter_name']
            values = observation.get('data')
            
            if values is None:
                print(f"Info: No data found for {parameter_name} at {station_name}")
                missing_data.append({
                    "Station": station_name,
                    "Parameter": parameter_name
                })
                continue
            
            if not isinstance(values, list):
                print(f"Warning: Unexpected data type for {parameter_name} at {station_name}. Expected list, got {type(values)}")
                continue

            for value in values:
                all_data.append({
                    "Station": station_name,
                    "Parameter": parameter_name,
                    "Timestamp": value.get("datetime"),
                    "Value": value.get("value")
                })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Handle missing data by adding columns with default value 0.0
    for missing in missing_data:
        station_name = missing['Station']
        parameter_name = missing['Parameter']
        column_name = f"{parameter_name} at {station_name}"
        
        # Create the column with default value 0.0 if it's missing
        if column_name not in df.columns:
            print(f"Adding missing column: {column_name} with default value 0.0")
            df[column_name] = 0.0
    
    # Create new column names with the format "Parameter at Station"
    df['Parameter_Station'] = df['Parameter'] + ' at ' + df['Station']
    
    # Pivot the table so each "Parameter_Station" becomes a column
    df_pivoted = df.pivot_table(index='Timestamp', columns='Parameter_Station', values='Value', aggfunc='first').reset_index()

    # For missing data columns, fill with 0.0
    for missing in missing_data:
        column_name = f"{missing['Parameter']} at {missing['Station']}"
        if column_name not in df_pivoted.columns:
            df_pivoted[column_name] = 0.0

    print("Pivoted DataFrame with missing data handled:")
    print(df_pivoted.head())
    print(f"\nTotal rows: {len(df_pivoted)}")

    # Create the save folder if it doesn't exist
    try:
        os.makedirs(SAVE_FOLDER, exist_ok=True)
    except PermissionError:
        print(f"Permission denied to create folder: {SAVE_FOLDER}")
        SAVE_FOLDER = os.path.expanduser('~')  # Default to home directory
        print(f"Defaulting to home directory: {SAVE_FOLDER}")

    # Generate a filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_{timestamp}.csv"
    filepath = os.path.join(SAVE_FOLDER, filename)

    # Save the DataFrame to a CSV file
    try:
        df_pivoted.to_csv(filepath, index=False)
        print(f"\nPivoted DataFrame saved to: {filepath}")
    except PermissionError:
        print(f"Permission denied to save file in {SAVE_FOLDER}")
        print("Please check your folder permissions and update the SAVE_FOLDER path.")

else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

# Debugging information
print("\nDebugging Information:")
print(f"Number of stations: {len(data)}")
for i, station in enumerate(data):
    print(f"Station {i+1}: {station['name']}")
    print(f"  Number of observations: {len(station['observations'])}")
    for j, observation in enumerate(station['observations']):
        print(f"    Observation {j+1}: {observation['parameter_name']}")
        print(f"      Data type: {type(observation.get('data'))}")
        print(f"      Data length: {len(observation.get('data', [])) if isinstance(observation.get('data'), list) else 'N/A'}")