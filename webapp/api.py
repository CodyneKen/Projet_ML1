import streamlit as st
import sounddevice as sd
import io
import numpy as np
import requests
import wavio
from datetime import datetime

# App Title
st.title("Détection d'émotion à partir de la voix")
st.write("Veuillez enregistrer votre voix pour que nous puissions détecter votre émotion")

# Recording Settings
SAMPLE_RATE = 44100  # CD quality
DURATION = 8  # seconds

recording = None

if st.button("Commencer l'enregistrement"):
    st.info(f"Enregistrement en cours... Veuillez parler pendant {DURATION} secondes.")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to finish
    st.success("Enregistrement terminé !")

    # Convert the recording to WAV in memory
    audio_buffer = io.BytesIO()
    wavio.write(audio_buffer, recording, SAMPLE_RATE, sampwidth=2)
    audio_buffer.seek(0)  # Reset pointer to the start

    # Playback audio
    st.audio(audio_buffer, format='audio/wav')


def predict_emotion(audio_data):
    st.info("Prédiction en cours...")

    # API endpoint
    api_url = "http://serving-api:8080/predict"
    files = {'file': ('audio.wav', audio_data, 'audio/wav')}

    try:
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Émotion détectée : {result.get('emotion', 'Inconnue')}")
        else:
            st.error("Erreur lors de la prédiction.")
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")


if recording is not None:
    if st.button("Prédire l'émotion"):
        predict_emotion(audio_buffer)

