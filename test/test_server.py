from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import logging
from typing import List

app = FastAPI()

# Set up logging with timestamp in milliseconds
logging.basicConfig(filename='predictions.log', level=logging.INFO, 
                    format='%(asctime)s.%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class Prediction(BaseModel):
    origin_code: str
    parameter_code: str
    time: datetime
    value: float

@app.post("/predictions")
async def receive_predictions(predictions: List[Prediction]):
    received_time = datetime.now()
    
    # Log separator for new batch of predictions
    logging.info("\n\n" + "="*40 + f"\nNew Batch of Predictions Received at {received_time.isoformat()}\n" + "="*40)

    # Iterate over each prediction in the list
    for prediction in predictions:
        log_message = (f"Received at {received_time.isoformat()} - "
                       f"Prediction time: {prediction.time.isoformat()} - "
                       f"Value: {prediction.value} - "
                       f"Origin: {prediction.origin_code} - "
                       f"Parameter: {prediction.parameter_code}")
        print(log_message)
        logging.info(log_message)
    
    return {"status": "success", "message": f"{len(predictions)} predictions received", "received_at": received_time.isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
