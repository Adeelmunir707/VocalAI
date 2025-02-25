import os
import torch
import whisper
import streamlit as st
from groq import Groq
from TTS.api import TTS
from tempfile import NamedTemporaryFile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import wave

# Function to get LLM response
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

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

# Function to generate speech from text
def generate_speech(text, output_file, speaker_wav, language="en", use_gpu=True):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav=speaker_wav,
        language=language,
    )

# Function to process audio frames for recording
def audio_callback(frame):
    audio = frame.to_ndarray()
    return av.AudioFrame.from_ndarray(audio, layout="mono")

def save_audio(frames, output_file):
    """Saves recorded audio frames to a .wav file."""
    sample_width = 2  # 16-bit PCM
    sample_rate = 44100
    channels = 1

    # Convert numpy frames to bytes
    audio_data = np.concatenate(frames, axis=0).astype(np.int16).tobytes()

    # Save the file
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

def main():
    st.set_page_config(page_title="Vocal AI", layout="wide")
    
    st.sidebar.title("Vocal-AI Settings")

    # Reference Audio (Upload or Record)
    ref_option = st.sidebar.radio("Reference Audio:", ("Upload", "Record"))
    reference_audio = None
    ref_audio_path = None

    if ref_option == "Upload":
        reference_audio = st.sidebar.file_uploader("Upload Reference Audio", type=["wav", "mp3", "ogg"])
    else:
        st.sidebar.write("Record Reference Audio:")
        webrtc_ctx_ref = webrtc_streamer(
            key="ref_audio",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True
        )

        if webrtc_ctx_ref.audio_receiver:
            ref_audio_frames = []
            while True:
                try:
                    frame = webrtc_ctx_ref.audio_receiver.get_frame(timeout=1)
                    ref_audio_frames.append(frame.to_ndarray())
                except:
                    break
            
            if ref_audio_frames:
                ref_audio_path = "ref_audio.wav"
                save_audio(ref_audio_frames, ref_audio_path)

    st.title("Welcome to VocaL AI")
    st.write("### How to Use")
    st.write("1. Upload or record a reference audio from the sidebar.")
    st.write("2. Choose between text input or audio input.")
    st.write("3. If audio input is selected, record or upload your audio.")
    st.write("4. Click 'Generate Speech' to hear the AI response in the cloned voice.")

    # User Input Audio (Upload or Record)
    input_type = st.radio("Choose Input Type", ("Text", "Audio"))
    user_input = None

    if input_type == "Text":
        user_input = st.text_area("Enter your text here")
    else:
        audio_option = st.radio("User Input Audio:", ("Upload", "Record"))
        user_audio_path = None

        if audio_option == "Upload":
            uploaded_audio = st.file_uploader("Upload Audio Input", type=["wav", "mp3", "ogg"])
            if uploaded_audio:
                with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(uploaded_audio.read())
                    user_audio_path = temp_audio.name
                user_input = transcribe_audio(user_audio_path)
                os.unlink(user_audio_path)

        else:
            st.write("Record User Audio:")
            webrtc_ctx_user = webrtc_streamer(
                key="user_audio",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": False, "audio": True},
                async_processing=True
            )

            if webrtc_ctx_user.audio_receiver:
                user_audio_frames = []
                while True:
                    try:
                        frame = webrtc_ctx_user.audio_receiver.get_frame(timeout=1)
                        user_audio_frames.append(frame.to_ndarray())
                    except:
                        break
                
                if user_audio_frames:
                    user_audio_path = "user_audio.wav"
                    save_audio(user_audio_frames, user_audio_path)
                    user_input = transcribe_audio(user_audio_path)

    if st.button("Generate Speech") and (reference_audio or ref_audio_path) and user_input:
        if ref_audio_path is None and reference_audio:
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
                temp_ref_audio.write(reference_audio.read())
                ref_audio_path = temp_ref_audio.name

        # Retrieve API Key from Streamlit secrets
        api_key = st.secrets["GROQ_API_KEY"]

        response_text = get_llm_response(api_key, user_input)
        output_audio_path = "output_speech.wav"
        generate_speech(response_text, output_audio_path, ref_audio_path)
        
        if os.path.exists(ref_audio_path):
            os.unlink(ref_audio_path)
        if user_audio_path and os.path.exists(user_audio_path):
            os.unlink(user_audio_path)

        st.audio(output_audio_path, format="audio/wav")

if __name__ == "__main__":
    main()
