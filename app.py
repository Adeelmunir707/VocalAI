import os
import torch
import whisper
import streamlit as st
from groq import Groq
from TTS.api import TTS
from tempfile import NamedTemporaryFile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import soundfile as sf

def get_llm_response(api_key, user_input):
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        top_p=1,
        stop=None,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def transcribe_audio(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

def generate_speech(text, output_file, speaker_wav, language="en", use_gpu=True):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav=speaker_wav,
        language=language,
    )

def save_audio(frames, sample_rate, output_file):
    audio = np.concatenate([np.frombuffer(frame.to_ndarray(), dtype=np.int16) for frame in frames])
    sf.write(output_file, audio, sample_rate)

st.set_page_config(page_title="Vocal AI", layout="wide")

st.sidebar.title("Vocal-AI Settings")

# Option for Reference Audio (Upload or Record)
ref_audio_option = st.sidebar.radio("Reference Audio", ("Upload", "Record"))

reference_audio = None
if ref_audio_option == "Upload":
    reference_audio = st.sidebar.file_uploader("Upload Reference Audio", type=["wav", "mp3", "ogg"])
else:
    st.sidebar.write("Record your reference audio:")
    ref_recorder = webrtc_streamer(
        key="ref_recorder",
        mode=WebRtcMode.RECVONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
    )

st.title("Welcome to VocaL AI")
st.write("### How to Use")
st.write("1. Choose a method for reference audio (upload or record).")
st.write("2. Select input type: text or audio.")
st.write("3. If audio input is selected, either upload or record your voice.")
st.write("4. Click 'Generate Speech' to hear the AI response in the cloned voice.")

# Option for User Input (Text or Audio)
input_type = st.radio("Choose Input Type", ("Text", "Audio"))

user_input = None
if input_type == "Text":
    user_input = st.text_area("Enter your text here")
else:
    audio_option = st.radio("User Input Audio", ("Upload", "Record"))
    
    if audio_option == "Upload":
        uploaded_audio = st.file_uploader("Upload Audio Input", type=["wav", "mp3", "ogg"])
        if uploaded_audio:
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(uploaded_audio.read())
                user_input = transcribe_audio(temp_audio.name)
            os.unlink(temp_audio.name)
    else:
        st.write("Record your input voice:")
        user_recorder = webrtc_streamer(
            key="user_recorder",
            mode=WebRtcMode.RECVONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"video": False, "audio": True},
        )

if st.button("Generate Speech") and user_input:
    if ref_audio_option == "Record" and ref_recorder and ref_recorder.audio_receiver:
        audio_frames = ref_recorder.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            ref_audio_path = "recorded_reference.wav"
            save_audio(audio_frames, 16000, ref_audio_path)
    elif reference_audio:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
            temp_ref_audio.write(reference_audio.read())
            ref_audio_path = temp_ref_audio.name

    api_key = st.secrets["GROQ_API_KEY"]
    
    response_text = get_llm_response(api_key, user_input)
    output_audio_path = "output_speech.wav"
    generate_speech(response_text, output_audio_path, ref_audio_path)
    
    st.audio(output_audio_path, format="audio/wav")
