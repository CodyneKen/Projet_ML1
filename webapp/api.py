import streamlit as st
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App Title
st.title("Détection d'émotion à partir de la voix")

# Description
st.write("Le but de cette application est de prédire l'émotion d'une personne à partir de sa voix. Vous pouvez uploader un fichier audio (.wav) pour effectuer la prédiction.")

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

if 'last_audio' not in st.session_state:
    st.session_state['last_audio'] = None

if 'feedback_sent' not in st.session_state:
    st.session_state['feedback_sent'] = False

# Emotion Mapping
emotion_mapping = {
    'Colère 😡​': 'C',
    'Tristesse 😢​': 'T',
    'Joie 😁​': 'J',
    'Peur 😨​': 'P',
    'Dégoût ​☹️​': 'D',
    'Surprise ​​😮​': 'S',
    'Neutre 😐​': 'N'
}

# Reverse mapping for feedback submission
reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

def predict_emotion(audio_data):
    api_url = "http://serving-api:8080/predict"
    files = {'file': ('audio.wav', audio_data, 'audio/wav')}
    try:
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction', 'Inconnue')
            st.session_state['last_prediction'] = prediction
            st.success(f"Émotion détectée : {prediction}")
        else:
            st.error(f"Erreur lors de la prédiction: {response.status_code}")
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")

def send_feedback(audio_data, predicted, correct):
    api_url = "http://serving-api:8080/feedback"
    files = {'file': ('audio.wav', audio_data, 'audio/wav')}
    data = {'prediction': predicted, 'target': correct}

    try:
        response = requests.post(api_url, files=files, data=data)
        if response.status_code == 200:
            st.session_state['feedback_sent'] = True
            st.success("Merci pour votre retour !")
        else:
            st.error(f"Erreur lors de l'envoi du feedback: {response.status_code}")
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API de feedback : {e}")

def feedback_section(audio_data, prediction):
    st.write("## Retour sur la prédiction")
    with st.form("feedback_form"):
        is_correct = st.radio("La prédiction est-elle correcte ?", ("Oui ✅", "Non ❌"))
        correct_emotion = None
        if is_correct == "Non ❌":
            correct_emotion = st.selectbox("Sélectionnez l'émotion correcte dans le cas où la prédiction est incorrecte :", options=list(emotion_mapping.keys()), index=None)
        submitted = st.form_submit_button("Envoyer le feedback")
        if submitted:
            if is_correct == "Oui ✅":
                prediction_code = emotion_mapping[prediction]
                send_feedback(audio_data, prediction_code, prediction_code)
            elif correct_emotion is not None:
                correct_emotion_code = emotion_mapping[correct_emotion]
                prediction_code = emotion_mapping[prediction]
                send_feedback(audio_data, prediction_code, correct_emotion_code)

uploaded_file = st.file_uploader("Uploader un fichier audio (.wav)", type=['wav'])
if uploaded_file is not None:
    st.success("Fichier chargé avec succès !")
    st.session_state['uploaded_file'] = uploaded_file
    st.session_state['last_audio'] = uploaded_file

if st.session_state['uploaded_file'] is not None:
    st.audio(st.session_state['uploaded_file'], format='audio/wav')
    if st.button("Prédire l'émotion"):
        predict_emotion(st.session_state['uploaded_file'].read())

if st.session_state['last_prediction'] is not None:
    feedback_section(st.session_state['last_audio'], st.session_state['last_prediction'])