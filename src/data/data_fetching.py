import os
import requests
import pandas as pd
from datetime import datetime

class DataFetcher:
    def __init__(self, base_url: str, save_folder: str = './data'):
        self.base_url = base_url
        self.save_folder = save_folder
        self.data = None
        self.df = None
        self.missing_data = []

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches data from the API and stores it as a DataFrame within the class.
        Returns the DataFrame.
        """
        response = requests.get(self.base_url)
        if response.status_code == 200:
            self.data = response.json()
            all_data = []

            for station in self.data:
                station_name = station['name']
                observations = station['observations']

                for observation in observations:
                    parameter_name = observation['parameter_name']
                    values = observation.get('data')

                    if values is None:
                        self.missing_data.append({
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

            # Create DataFrame from all_data
            self.df = pd.DataFrame(all_data)
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None

        return self.df  # Return the DataFrame

    def handle_missing_data(self) -> pd.DataFrame:
        """
        Handles missing data by adding missing columns with default value 0.0,
        and pivots the DataFrame to have "Parameter at Station" as columns.
        Returns the pivoted DataFrame.
        """
        if self.df is None:
            print("Data has not been fetched yet.")
            return None

        # Create new column names with the format "Parameter at Station"
        self.df['Parameter_Station'] = self.df['Parameter'] + ' at ' + self.df['Station']

        # Pivot the table so each "Parameter_Station" becomes a column
        df_pivoted = self.df.pivot_table(index='Timestamp', columns='Parameter_Station', values='Value', aggfunc='first').reset_index()

        # For missing data columns, fill with 0.0
        for missing in self.missing_data:
            column_name = f"{missing['Parameter']} at {missing['Station']}"
            if column_name not in df_pivoted.columns:
                print(f"Adding missing column: {column_name} with default value 0.0")
                df_pivoted[column_name] = 0.0

        self.df = df_pivoted  # Update the internal df to the pivoted version
        return self.df  # Return the pivoted DataFrame

    def save_dataframe(self) -> str:
        """
        Saves the DataFrame to a CSV file in the specified folder.

        :return: The file path of the saved CSV file.
        """
        if self.df is None:
            print("Data has not been processed yet.")
            return None

        try:
            os.makedirs(self.save_folder, exist_ok=True)
        except PermissionError:
            print(f"Permission denied to create folder: {self.save_folder}")
            self.save_folder = os.path.expanduser('~')  # Default to home directory
            print(f"Defaulting to home directory: {self.save_folder}")

        # Generate a filename with current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.csv"
        filepath = os.path.join(self.save_folder, filename)

        # Save the DataFrame to a CSV file
        try:
            self.df.to_csv(filepath, index=False)
            print(f"\nDataFrame saved to: {filepath}")
        except PermissionError:
            print(f"Permission denied to save file in {self.save_folder}")
            print("Please check your folder permissions and update the SAVE_FOLDER path.")

        return filepath
