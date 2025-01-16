from fastapi import FastAPI, File, UploadFile
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
    data, sample_rate = librosa.load(audio_stream, duration=2.5, offset=0.6)
  except Exception as e:
    return {"error": f"Failed to process the audio file: {str(e)}"}
  
  # Return the predicted label
  return fct_model.predict_on_audio(model, encoder, scaler, data, sample_rate)