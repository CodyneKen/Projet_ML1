import streamlit as st
import sounddevice as sd
import wavio
import numpy as np
import os
from datetime import datetime

# App Title
st.title("Détection d'émotion à partir de la voix")
st.write('Veuillez enregistrer votre voix pour que nous puissions détecter votre émotion')

# Recording Settings
SAMPLE_RATE = 44100  # CD quality
DURATION = 8  # seconds

if st.button("Commencer l'enregistrement"):
    st.info(f"Recording for {DURATION} seconds...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to finish
    st.success("Enregistrement terminé !")

    output_dir = './Projet_ML1/data/raw/received_audio'
    os.makedirs(output_dir, exist_ok=True)

    # Save as WAV file
    filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    file_path = os.path.join(output_dir, filename)
    wavio.write(file_path, recording, SAMPLE_RATE, sampwidth=2)

    # st.success(f"Audio saved as {file_path}")
    st.audio(file_path, format='audio/wav')
