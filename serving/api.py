from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
#import dill
#import numpy as np
import librosa
import io

import sys
import fct_model

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

encoder = joblib.load("/artifacts/encoder.pkl")
model = joblib.load("/artifacts/model.pkl")
scaler = joblib.load("/artifacts/scaler.pkl")
prod_path = "/data/prod_data.csv"

@app.get("/")
def read_root(input):
  return {"message": f"Hello, {input}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  """
    Predict endpoint to handle audio file uploads and make predictions.
    - file: Uploaded .wav file.
  """
  logger.info("Received file: %s", file.filename)

  # Load the audio file using librosa
  try:
    audio_bytes = await file.read()
    audio_stream = io.BytesIO(audio_bytes)
    data, sample_rate = librosa.load(audio_stream, duration=7, offset=0.2)
  except Exception as e:
    return {"error": f"Failed to process the audio file: {str(e)}"}
  
  # Return the predicted label
  return fct_model.predict_on_audio(model, encoder, scaler, data, sample_rate)

@app.post("/feedback")
async def feedback(prediction, target, file: UploadFile = File(...)):
  """
  Send feedback of model's prediction.
  Feedback is then saved in /data/prod_data.csv with embedding, target, and prediction.
  
  Args:
  - prediction (string): 'C' or 'T' or 'J' or 'P' or 'D' or 'S' or 'N'
  - target (string): 'C' or 'T' or 'J' or 'P' or 'D' or 'S' or 'N'
  - file (UploadFile)
  """
  logger.info("Received file: %s", file.filename)
  logger.info("Received prediction: %s", prediction)
  logger.info("Received target: %s", target)
  
  try:
    audio_bytes = await file.read()
    audio_stream = io.BytesIO(audio_bytes)
    data, sample_rate = librosa.load(audio_stream, duration=7, offset=0.2)
  except Exception as e:
    return {"error": f"Failed to process the audio file: {str(e)}"}
  
  # Save feedback in prod_data
  fct_model.save_feedback(data, sample_rate, target, prediction, prod_path)
  
  return JSONResponse(
    content={"message": "Merci pour votre retour !"},
    status_code=200
  )
    
  
  