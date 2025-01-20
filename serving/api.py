from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import librosa
import io

import fct_model

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
k = 1
deploy_only_best_acc = False

artifacts_path = "/artifacts/"
ref_path = "/data/ref_data.csv"
prod_path = "/data/prod_data.csv"

encoder = joblib.load("/artifacts/encoder.pkl")
model_data = joblib.load("/artifacts/model.pkl")
model = model_data["model"]
last_accuracy = model_data["last_accuracy"]
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
    data, sample_rate = librosa.load(audio_stream, duration=7, offset=0.2)
  except Exception as e:
    return {"error": f"Failed to process the audio file: {str(e)}"}
  
  # Return the predicted label
  logger.info("Model accuracy: %s", last_accuracy)
  return fct_model.predict_on_audio(model, encoder, scaler, data, sample_rate)

@app.post("/feedback")
async def feedback(background_tasks: BackgroundTasks, prediction, target, file: UploadFile = File(...)):
  global model, last_accuracy
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
  
  # Check if model needs to be updated and update in a non-blocking manner
  #if not hasattr(feedback, "task_added") or not feedback.task_added:
  #  feedback.task_added = True
  background_tasks.add_task(retrain_model)

  return JSONResponse(
    content={"message": "Merci pour votre retour !"},
    status_code=200
  )
    
def retrain_model():
  global model, last_accuracy, k, prod_path
  # Check if model needs to be updated
  if fct_model.should_retrain_model(k, prod_path):
    logger.info("\nLast accuracy: %s", last_accuracy)
    model_full, history = fct_model.train_save_model(ref_path, artifacts_path, 1, prod_path, not(deploy_only_best_acc), last_accuracy)
    new_accuracy = history.history['accuracy'][-1]
    if deploy_only_best_acc:
      if new_accuracy > last_accuracy:
        logger.info("New accuracy (%s) is better than last accuracy (%s). Deploying new model.", new_accuracy, last_accuracy)
        model = model_full
        last_accuracy = new_accuracy
      else:
        logger.info("New accuracy (%s) is worse than last accuracy (%s). Not deploying model.", new_accuracy, last_accuracy)
    else:
      # In case deploy_only_best_acc is False, always deploy the new model
      logger.info("Deploying new model regardless of accuracy. New accuracy: %s, Last accuracy: %s", new_accuracy, last_accuracy)
      model = model_full
      last_accuracy = new_accuracy
  else:
    logger.info("Hoho")