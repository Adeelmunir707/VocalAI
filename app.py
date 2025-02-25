import os
import torch
import whisper
import streamlit as st
from groq import Groq
from TTS.api import TTS
from tempfile import NamedTemporaryFile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# LLM Response Function
def get_llm_response(api_key, user_input):
    client = Groq(api_key=api_key)
    prompt = (
        "IMPORTANT: You are an AI assistant that MUST provide responses in 25 words or less.\n"
        "CRITICAL RULES:\n"
        "1. NEVER exceed 25 words unless absolutely necessary.\n"
        "2. Always give a complete sentence with full context.\n"
        "3. Answer directly and precisely.\n"
        "4. Use clear, simple language.\n"
        "5. Maintain a polite, professional tone.\n"
        "6. NO lists, bullet points, or multiple paragraphs.\n"
        "7. NEVER apologize for brevity - embrace it.\n"
        "Your response will be converted to speech. Maximum 25 words."
    )
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        top_p=1,
        stream=False,
    )
    return chat_completion.choices[0].message.content

# Transcribe Audio
def transcribe_audio(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

# Generate Speech
def generate_speech(text, output_file, speaker_wav, language="en", use_gpu=True):
    if not os.path.exists(speaker_wav):
        raise FileNotFoundError("Reference audio file not found. Please upload or record a valid audio.")
    
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav=speaker_wav,
        language=language,
    )

# Streamlit App
def main():
    st.set_page_config(page_title="Vocal AI", layout="wide")
    st.sidebar.title("Vocal-AI Settings")

    # User option for reference audio (Record or Upload)
    ref_audio_choice = st.sidebar.radio("Reference Audio", ("Upload", "Record"))

    reference_audio = None
    ref_audio_path = None

    if ref_audio_choice == "Upload":
        reference_audio = st.sidebar.file_uploader("Upload Reference Audio", type=["wav", "mp3", "ogg"])
    else:
        st.sidebar.write("Record your reference audio:")
        ref_audio_recorder = webrtc_streamer(
            key="ref_audio",
            mode=WebRtcMode.SENDONLY,
            client_settings=ClientSettings(rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        )

    st.title("Welcome to VocaL AI")
    st.write("### How to Use")
    st.write("1. Upload or record a reference audio file.")
    st.write("2. Choose between text or audio input.")
    st.write("3. If audio input is selected, record and submit your audio.")
    st.write("4. Click 'Generate Speech' to hear the AI response in your cloned voice.")

    # User Input (Text or Audio)
    input_type = st.radio("Choose Input Type", ("Text", "Audio"))
    user_input = None

    if input_type == "Text":
        user_input = st.text_area("Enter your text here")
    else:
        st.write("Record your voice:")
        user_audio_recorder = webrtc_streamer(
            key="user_audio",
            mode=WebRtcMode.SENDONLY,
            client_settings=ClientSettings(rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        )

    if st.button("Generate Speech"):
        # Handle Reference Audio
        if reference_audio:
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
                temp_ref_audio.write(reference_audio.read())
                ref_audio_path = temp_ref_audio.name
        elif ref_audio_recorder and ref_audio_recorder.audio_receiver:
            ref_audio_path = ref_audio_recorder.audio_receiver.get_audio_frame()

        if not ref_audio_path:
            st.error("Please upload or record reference audio.")
            return

        # Handle User Input
        if input_type == "Audio" and user_audio_recorder and user_audio_recorder.audio_receiver:
            user_input_audio = user_audio_recorder.audio_receiver.get_audio_frame()
            if user_input_audio:
                with NamedTemporaryFile(delete=False, suffix=".wav") as temp_user_audio:
                    temp_user_audio.write(user_input_audio)
                    user_input = transcribe_audio(temp_user_audio.name)
                os.unlink(temp_user_audio.name)

        if not user_input:
            st.error("Please enter text or record audio.")
            return

        # Get AI Response
        api_key = st.secrets["GROQ_API_KEY"]
        response_text = get_llm_response(api_key, user_input)

        # Generate Speech
        output_audio_path = "output_speech.wav"
        try:
            generate_speech(response_text, output_audio_path, ref_audio_path)
            os.unlink(ref_audio_path)
            st.audio(output_audio_path, format="audio/wav")
        except FileNotFoundError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
