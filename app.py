import streamlit as st
import tempfile
import os
import torch
import whisper
from TTS.api import TTS
from groq import Groq
from pydub import AudioSegment
from streamlit_option_menu import option_menu

# Hardcoded Groq API Key
GROQ_API_KEY = "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"

# Load TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

def get_llm_response(user_input):
    client = Groq(api_key=GROQ_API_KEY)
    prompt = ("IMPORTANT: You are an AI assistant that MUST provide responses in 25 words or less. NO EXCEPTIONS...")
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
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

def generate_speech(text, output_file, speaker_wav, language="hi"):
    tts.tts_to_file(text=text, file_path=output_file, speaker_wav=speaker_wav, language=language)

# UI Design
st.title("🗣️ VocalAI - AI-Powered Voice Cloning & Chatbot")
selected_page = option_menu(
    menu_title=None,
    options=["Text-to-Speech", "Voice-Cloned Chatbot"],
    icons=["mic", "chat-dots"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Sidebar - Reference Speaker Audio Upload
st.sidebar.header("Upload Reference Audio")
ref_audio = st.sidebar.file_uploader("Upload a speaker audio file (WAV format)", type=["wav", "ogg", "mp3"])

if selected_page == "Text-to-Speech":
    st.header("🔊 Text-to-Speech (TTS)")
    text = st.text_area("Enter text to synthesize:", "Hello, this is a cloned voice test.")
    
    if st.button("Generate Voice"):
        if ref_audio is None:
            st.warning("⚠️ Please upload a reference speaker audio file first!")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
                temp_speaker.write(ref_audio.read())
                speaker_wav_path = temp_speaker.name
            
            output_path = "cloned_output.wav"
            generate_speech(text, output_path, speaker_wav_path, language="en")
            st.audio(output_path, format="audio/wav")
            
            with open(output_path, "rb") as f:
                st.download_button("Download Cloned Voice", f, file_name="cloned_voice.wav", mime="audio/wav")
            
            os.unlink(speaker_wav_path)

elif selected_page == "Voice-Cloned Chatbot":
    st.header("💬 AI Chatbot with Voice Cloning")
    user_query = st.text_area("Enter your query:", "Hello, explain AI briefly.")
    uploaded_voice = st.file_uploader("Or upload an audio query (WAV format)", type=["wav", "ogg", "mp3"])
    
    if uploaded_voice is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_voice.read())
            audio_path = temp_audio.name
        user_query = transcribe_audio(audio_path)
        os.unlink(audio_path)
        st.write("**Transcribed Query:**", user_query)
    
    if st.button("Generate Response"):
        if ref_audio is None:
            st.warning("⚠️ Please upload a reference speaker audio file first!")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
                temp_speaker.write(ref_audio.read())
                speaker_wav_path = temp_speaker.name
            
            response = get_llm_response(user_query)
            output_audio_path = "cloned_chat_response.wav"
            generate_speech(response, output_audio_path, speaker_wav_path, language="en")
            
            st.audio(output_audio_path, format="audio/wav")
            
            with open(output_audio_path, "rb") as f:
                st.download_button("Download Response Audio", f, file_name="cloned_chat_response.wav", mime="audio/wav")
            
            os.unlink(speaker_wav_path)
