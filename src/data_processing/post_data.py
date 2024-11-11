import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd

def send_json_to_url_with_wait(json_data, url, headers=None, wait_time=5):
    json_payload = json.dumps(json_data)
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, data=json_payload, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        print(f"Request successful: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    # Wait for the specified wait time before returning
    time.sleep(wait_time)

def send_predictions(df, url, origin_code, parameter_code):
    for idx, row in df.iterrows():
        base_time = datetime.strptime(str(idx), "%Y-%m-%d %H:%M:%S")
        
        # Create a list of JSON objects for the predictions of the current row
        json_data_list = []
        for i in range(1, 5):  # t+1 to t+4 predictions
            prediction_time = base_time + timedelta(hours=i)
            prediction_value = row[f't+{i} prediction']
            
            json_data_list.append({
                "origin_code": origin_code,
                "parameter_code": parameter_code,
                "time": prediction_time.isoformat(),
                "value": float(prediction_value)
            })
        
        # Send the entire list of JSON objects in one request
        send_json_to_url_with_wait(json_data_list, url)
        print(f"Sent predictions for {base_time}")