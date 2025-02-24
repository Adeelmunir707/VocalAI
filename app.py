import os
import torch
import whisper
import streamlit as st
from groq import Groq
from TTS.api import TTS

# Load API key from Streamlit secrets
def get_api_key():
    return st.secrets["GROQ_API_KEY"]

# LLM Response Function
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
    response = chat_completion.choices[0].message.content
    return response

# Transcribe Audio File using Whisper
def transcribe_audio(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

# Generate Speech from Text
def generate_speech(text, output_file, speaker_wav, language="en", use_gpu=True):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav=speaker_wav,
        language=language,
    )

# Streamlit UI
st.title("🎙️ AI Voice Assistant")
st.markdown("Upload an audio file or enter text, and the AI will generate a response.")

api_key = get_api_key()

# Input Selection
input_type = st.radio("Choose Input Type:", ["Text", "Audio"])

if input_type == "Text":
    user_text = st.text_area("Enter text:")
    if st.button("Generate Response"):
        if user_text.strip():
            response = get_llm_response(api_key, user_text)
            st.subheader("📝 AI Response")
            st.write(response)
            
            # TTS Generation
            speaker_wav = "reference_voice.ogg"  # Replace with a valid speaker sample
            output_audio_path = "output_response.wav"
            generate_speech(response, output_audio_path, speaker_wav)
            
            st.audio(output_audio_path, format="audio/wav")
        else:
            st.error("Please enter some text.")

elif input_type == "Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
    if audio_file and st.button("Transcribe & Generate Response"):
        with open("input_audio.wav", "wb") as f:
            f.write(audio_file.read())

        # Transcribe
        st.info("Transcribing audio...")
        transcribed_text = transcribe_audio("input_audio.wav")
        st.subheader("🎤 Transcription")
        st.write(transcribed_text)

        # Get AI Response
        response = get_llm_response(api_key, transcribed_text)
        st.subheader("📝 AI Response")
        st.write(response)
        
        # TTS Generation
        speaker_wav = "reference_voice.ogg"  # Replace with a valid speaker sample
        output_audio_path = "output_response.wav"
        generate_speech(response, output_audio_path, speaker_wav)

        st.audio(output_audio_path, format="audio/wav")
