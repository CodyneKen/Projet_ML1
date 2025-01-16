import streamlit as st
import sounddevice as sd
import io
import numpy as np
import requests
import wavio
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App Title
st.title("Détection d'émotion à partir de la voix")

# Description
st.write("Le but de cette application est de prédire l'émotion d'une personne à partir de sa voix. Vous pouvez enregistrer votre voix ou uploader un fichier audio (.wav) pour effectuer la prédiction.")

# Recording Settings
SAMPLE_RATE = 44100  # CD quality
DURATION = 8  # seconds

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state['recording'] = None

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

def predict_emotion(audio_data):
    st.info("Prédiction en cours...")

    api_url = "http://serving-api:8080/predict"

    files = {'file': ('audio.wav', audio_data, 'audio/wav')}

    try:
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            result = response.json()
            logger.info("Received response: %s", result)
            st.success(f"Émotion détectée : {result.get('prediction', 'Inconnue')}")
        else:
            st.error(f"Erreur lors de la prédiction: {response.status_code}")
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")



st.subheader("S'enregistrer (bug car docker ne trouve pas le device audio)")
# Recording Button
if st.button("Commencer l'enregistrement"):
    st.info(f"Enregistrement en cours... Veuillez parler pendant {DURATION} secondes.")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    st.success("Enregistrement terminé !")

    # Convert the recording to WAV in memory
    audio_buffer = io.BytesIO()
    wavio.write(audio_buffer, recording, SAMPLE_RATE, sampwidth=2)
    audio_buffer.seek(0)

    st.session_state['recording'] = audio_buffer

# Playback recorded audio
if st.session_state['recording'] is not None:
    st.audio(st.session_state['recording'], format='audio/wav')
    if st.button("Prédire l'émotion (Enregistrement)"):
        predict_emotion(st.session_state['recording'].read())

st.subheader("Avec un fichier audio existant")
# File Uploader
uploaded_file = st.file_uploader("Uploader un fichier audio (.wav)", type=['wav'])
if uploaded_file is not None:
    st.success("Fichier chargé avec succès !")
    st.session_state['uploaded_file'] = uploaded_file

# Playback uploaded audio
if st.session_state['uploaded_file'] is not None:
    st.audio(st.session_state['uploaded_file'], format='audio/wav')
    if st.button("Prédire l'émotion (Fichier Uploadé)"):
        predict_emotion(st.session_state['uploaded_file'].read())
