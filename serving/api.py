from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import dill
import numpy as np
import librosa
import io

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

encoder = joblib.load("/artifacts/encoder.pkl")
model = joblib.load("/artifacts/model.pkl")
scaler = joblib.load("/artifacts/scaler.pkl")
with open("/artifacts/extract_features.pkl", "rb") as f:
  get_embedding = dill.load(f)
  get_embedding.__globals__["np"] = np
  get_embedding.__globals__["librosa"] = librosa

emotion_mapping = {
  'C': 'anger',
  'T': 'sadness',
  'J': 'joy',
  'P': 'fear',
  'D': 'disgust',
  'S': 'surprise',
  'N': 'neutral'
}

@app.get("/")
def read_root(input):
  return {"message": f"Hello, {input}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  """
    Predict endpoint to handle audio file uploads and make predictions.
    - file: Uploaded .wav file.
  """
  embedding_size = 162
  logger.info("Received file: %s", file.filename)
  
  # Save the uploaded file to a temporary location
  #file_location = f"/tmp/{file.filename}"
  #with open(file_location, "wb") as f:
  #  f.write(await file.read())

  # Load the audio file using librosa
  try:
    audio_bytes = await file.read()
    audio_stream = io.BytesIO(audio_bytes)
    data, sample_rate = librosa.load(audio_stream, duration=2.5, offset=0.6)
  except Exception as e:
    return {"error": f"Failed to process the audio file: {str(e)}"}

  # Extract features using the pre-loaded function
  try:
    features = get_embedding(data, sample_rate)
  except Exception as e:
    return {"error": f"Failed to extract features: {str(e)}"}

  # Scale and reshape the features for the model
  features_scaled = scaler.transform(features.reshape(1, -1))
  features_reshaped = features_scaled.reshape(1, embedding_size, 1)  # Ensure the shape matches the model input

  # Predict the class probabilities
  try:
    pred = model.predict(features_reshaped)
    predicted_class_label = encoder.inverse_transform(pred)[0][0]
    predicted_emotion = emotion_mapping[predicted_class_label]
  except Exception as e:
    return {"error": f"Prediction failed: {str(e)}"}

  # Return the predicted label
  return {"prediction": predicted_emotion}
  