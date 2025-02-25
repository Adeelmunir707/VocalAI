import os
import torch
import whisper
import streamlit as st
from groq import Groq
from TTS.api import TTS
from tempfile import NamedTemporaryFile
from google.colab import userdata

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

def main():
    st.set_page_config(page_title="VocaGenie", layout="wide")
    
    st.sidebar.title("VocaGenie Settings")
    reference_audio = st.sidebar.file_uploader("Upload Reference Audio", type=["wav", "mp3", "ogg"])
    
    st.title("Welcome to VocaGenie")
    st.write("### How to Use")
    st.write("1. Upload a reference audio file from the sidebar.")
    st.write("2. Choose between text input or audio input.")
    st.write("3. If audio input is selected, record and submit your audio.")
    st.write("4. Click the 'Generate Speech' button to hear the AI response in the cloned voice.")
    
    input_type = st.radio("Choose Input Type", ("Text", "Audio"))
    user_input = None
    
    if input_type == "Text":
        user_input = st.text_area("Enter your text here")
    else:
        uploaded_audio = st.file_uploader("Upload Audio Input", type=["wav", "mp3", "ogg"])
        if uploaded_audio:
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(uploaded_audio.read())
                user_input = transcribe_audio(temp_audio.name)
            os.unlink(temp_audio.name)
    
    if st.button("Generate Speech") and reference_audio and user_input:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
            temp_ref_audio.write(reference_audio.read())
            ref_audio_path = temp_ref_audio.name
        
        api_key = userdata.get('groq')
        response_text = get_llm_response(api_key, user_input)
        output_audio_path = "output_speech.wav"
        generate_speech(response_text, output_audio_path, ref_audio_path)
        os.unlink(ref_audio_path)
        
        st.audio(output_audio_path, format="audio/wav")

if __name__ == "__main__":
    main()
