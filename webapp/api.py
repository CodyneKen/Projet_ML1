import streamlit as st

#For mic recording
from streamlit_mic_recorder import mic_recorder

#To save audio file
import os
from datetime import datetime
import wave


st.title('Welcome to the ML1 project!')

st.write('This is the webapp for the ML1 project. You can use it to make predictions on the dataset.')



#Features a rajouter :
# - User can send audio data and/or record audio data
#RECORD
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None
)

# Save .wav file in received_audio folder
if audio:
    # st.audio(audio, format='audio/wav')
    # Create directory if it doesn't exist
    output_dir = './Projet_ML1/data/raw/received_audio'
    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename
    filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    file_path = os.path.join(output_dir, filename)

    # Save the audio file
    FRAMES_PER_SECOND = 44100

    with wave.open(file_path, mode="wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(4)
        wav_file.setframerate(FRAMES_PER_SECOND)
        wav_file.writeframes(audio['bytes'])
    
    st.success(f"WARNING : DO NOT LISTEN TO THE AUDIO, WILL BLOW YOUR EARDRUMS. Audio file saved as {file_path}")
    audio = None




# - User can click a button to trigger the Serving api and prediction (python "requests" package)
# - Prediction is displayed
# - Champ de saisie et bouton feedback pour reporting (entrer la cible réelle)