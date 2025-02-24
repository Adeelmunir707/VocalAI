import os
import streamlit as st
import torch
import whisper
from groq import Groq
from TTS.api import TTS

# Function to get LLM response from Groq
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

# Function to transcribe audio
def transcribe_audio(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

# Function to generate speech from text
def generate_speech(text, output_file, speaker_wav, language="en", use_gpu=True):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    tts.tts_to_file(text=text, file_path=output_file, speaker_wav=speaker_wav, language=language)

# Streamlit UI
st.title("🎤 LLM-Powered Voice Assistant")
st.markdown("This app transcribes audio, generates responses using Llama 3, and converts text to speech.")

# User input options
input_choice = st.radio("Choose Input Type:", ("Text", "Audio"))

api_key = st.text_input("Enter Groq API Key:", type="password")

if input_choice == "Text":
    user_input = st.text_area("Enter text:", placeholder="Type your question or statement here...")
    uploaded_audio = None
else:
    user_input = None
    uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "ogg", "m4a"])

# Speaker voice file for XTTS v2
speaker_wav = st.file_uploader("Upload a reference voice sample (optional):", type=["wav", "mp3", "ogg"])

if st.button("Process"):
    if not api_key:
        st.error("Please enter the API key!")
    elif not user_input and not uploaded_audio:
        st.error("Please provide either text input or an audio file!")
    else:
        with st.spinner("Processing..."):
            if uploaded_audio:
                temp_audio_path = "temp_audio.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_audio.read())
                
                st.write("🎙 Transcribing Audio...")
                transcribed_text = transcribe_audio(temp_audio_path)
                st.success(f"Transcription: {transcribed_text}")
                user_input = transcribed_text  # Use transcribed text as input
            
            # Get LLM response
            response = get_llm_response(api_key, user_input)
            st.success(f"🤖 LLM Response: {response}")

            # Generate speech
            output_audio_path = "output_speech.wav"
            if speaker_wav:
                speaker_wav_path = "speaker_ref.wav"
                with open(speaker_wav_path, "wb") as f:
                    f.write(speaker_wav.read())
            else:
                speaker_wav_path = None
            
            generate_speech(response, output_audio_path, speaker_wav_path)
            st.audio(output_audio_path, format="audio/wav")
            st.success("🔊 Speech generated successfully! Download below:")
            st.download_button("Download Speech", data=open(output_audio_path, "rb"), file_name="response.wav", mime="audio/wav")
