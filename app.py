# import streamlit as st
# import tempfile
# import os
# import torch
# import whisper
# from TTS.api import TTS
# from groq import Groq
# from pydub import AudioSegment
# from streamlit_option_menu import option_menu

# # Hardcoded Groq API Key
# GROQ_API_KEY = "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"

# # Load TTS model
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# def get_llm_response(user_input):
#     client = Groq(api_key=GROQ_API_KEY)
#     prompt = ("IMPORTANT: You are an AI assistant that MUST provide responses in 25 words or less. NO EXCEPTIONS...")
    
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": user_input}
#         ],
#         model="llama3-8b-8192",
#         temperature=0.5,
#         top_p=1,
#         stop=None,
#         stream=False,
#     )
#     return chat_completion.choices[0].message.content

# def transcribe_audio(audio_path, model_size="base"):
#     model = whisper.load_model(model_size)
#     result = model.transcribe(audio_path)
#     return result["text"]

# def generate_speech(text, output_file, speaker_wav, language="hi"):
#     with torch.inference_mode():
#         tts.tts_to_file(text=text, file_path=output_file, speaker_wav=speaker_wav, language=language)

# # UI Design
# st.title("🗣️ VocalAI - AI-Powered Voice Cloning & Chatbot")
# selected_page = option_menu(
#     menu_title=None,
#     options=["Text-to-Speech", "Voice-Cloned Chatbot"],
#     icons=["mic", "chat-dots"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal"
# )

# # Sidebar - Reference Speaker Audio Upload
# st.sidebar.header("Upload Reference Audio")
# ref_audio = st.sidebar.file_uploader("Upload a speaker audio file (WAV format)", type=["wav", "ogg", "mp3"])

# if selected_page == "Text-to-Speech":
#     st.header("🔊 Text-to-Speech (TTS)")
#     text = st.text_area("Enter text to synthesize:", "Hello, this is a cloned voice test.")
    
#     if st.button("Generate Voice"):
#         if ref_audio is None:
#             st.warning("⚠️ Please upload a reference speaker audio file first!")
#         else:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
#                 temp_speaker.write(ref_audio.read())
#                 speaker_wav_path = temp_speaker.name
            
#             output_path = "cloned_output.wav"
#             generate_speech(text, output_path, speaker_wav_path, language="en")
#             st.audio(output_path, format="audio/wav")
            
#             with open(output_path, "rb") as f:
#                 st.download_button("Download Cloned Voice", f, file_name="cloned_voice.wav", mime="audio/wav")
            
#             os.unlink(speaker_wav_path)

# # elif selected_page == "Voice-Cloned Chatbot":
# #     st.header("💬 AI Chatbot with Voice Cloning")
# #     user_query = st.text_area("Enter your query:", "Hello, e  xplain AI briefly.")
# #     uploaded_voice = st.file_uploader("Or upload an audio query (WAV format)", type=["wav", "ogg", "mp3"])
    
# #     if uploaded_voice is not None:
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
# #             temp_audio.write(uploaded_voice.read())
# #             audio_path = temp_audio.name
# #         user_query = transcribe_audio(audio_path)
# #         os.unlink(audio_path)
# #         st.write("**Transcribed Query:**", user_query)
    
# #     if st.button("Generate Response"):
# #         if ref_audio is None:
# #             st.warning("⚠️ Please upload a reference speaker audio file first!")
# #         else:
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
# #                 temp_speaker.write(ref_audio.read())
# #                 speaker_wav_path = temp_speaker.name
            
# #             response = get_llm_response(user_query)
# #             output_audio_path = "cloned_chat_response.wav"
# #             generate_speech(response, output_audio_path, speaker_wav_path, language="hi")
            
# #             st.audio(output_audio_path, format="audio/wav")
            
# #             with open(output_audio_path, "rb") as f:
# #                 st.download_button("Download Response Audio", f, file_name="cloned_chat_response.wav", mime="audio/wav")
            
# #             os.unlink(speaker_wav_path)


# elif selected_page == "Voice-Cloned Chatbot":
#     st.header("💬 AI Chatbot with Voice Cloning")

#     # Check if reference audio is uploaded
#     if ref_audio is None:
#         st.warning("⚠️ Please upload a reference speaker audio file first!")
    
#     # Input Type Selection
#     input_type = st.radio("Choose input type:", ["Text", "Audio"])

#     user_query = ""
#     uploaded_voice = None

#     # Text Input Mode
#     if input_type == "Text":
#         user_query = st.text_area("Enter your query:", "Hello, explain AI briefly.")

#     # Audio Input Mode
#     elif input_type == "Audio":
#         uploaded_voice = st.file_uploader("Upload an audio query (WAV format)", type=["wav", "ogg", "mp3"])
        
#         if uploaded_voice is not None:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#                 temp_audio.write(uploaded_voice.read())
#                 audio_path = temp_audio.name

#             # Transcribe Audio
#             user_query = transcribe_audio(audio_path)
#             os.unlink(audio_path)  # Delete audio file after transcription
            
#             # Display transcribed text for user confirmation
#             st.text_area("Transcribed Query:", user_query, disabled=False   )

#     # Generate Response
#     if st.button("Generate Response", disabled=ref_audio is None or not user_query.strip()):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
#             temp_speaker.write(ref_audio.read())
#             speaker_wav_path = temp_speaker.name
        
#         # Get AI Response
#         response = get_llm_response(user_query)
        
#         # Convert AI Response to Speech
#         output_audio_path = "cloned_chat_response.wav"
#         generate_speech(response, output_audio_path, speaker_wav_path, language="en")

#         # Play & Download Response Audio
#         st.audio(output_audio_path, format="audio/wav")
#         with open(output_audio_path, "rb") as f:
#             st.download_button("Download Response Audio", f, file_name="cloned_chat_response.wav", mime="audio/wav")

#         # Cleanup
#         os.unlink(speaker_wav_path)


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
    prompt = ("IMPORTANT: You are an AI assistant that MUST provide responses in 25 words or less. NO EXCEPTIONS. "
              "CRITICAL RULES:\n"
              "1. NEVER exceed 25 words in your response only in exceptional cases where it is required strictly \n"
              "2. Always give a complete sentence with full context\n"
              "3. Answer directly and precisely what is asked\n"
              "4. Use simple, clear language appropriate for voice\n"
              "5. Maintain polite, professional tone\n"
              "6. NEVER provide lists, bullet points, or numbered items\n"
              "7. NEVER write multiple paragraphs\n"
              "8. NEVER apologize for brevity - embrace it\n\n"
              "REMEMBER: Your responses will be converted to speech. Exactly ONE brief paragraph. Maximum 25 words providing full contextual understanding.")
    
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

def transcribe_audio(audio_path, model_size="large"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

def generate_speech(text, output_file, speaker_wav, language):
    with torch.inference_mode():
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

# Sidebar - Language Selection & Reference Speaker Audio Upload
st.sidebar.header("Settings")
language = st.sidebar.selectbox("Select Language", ["English", "Hindi"], index=0)
st.sidebar.header("Upload Reference Audio")
ref_audio = st.sidebar.file_uploader("Upload a speaker audio file (WAV format)", type=["wav", "ogg", "mp3"])

language_code = "en" if language == "English" else "hi"

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
            generate_speech(text, output_path, speaker_wav_path, language_code)
            st.audio(output_path, format="audio/wav")
            
            with open(output_path, "rb") as f:
                st.download_button("Download Cloned Voice", f, file_name="cloned_voice.wav", mime="audio/wav")
            
            os.unlink(speaker_wav_path)

elif selected_page == "Voice-Cloned Chatbot":
    st.header("💬 AI Chatbot with Voice Cloning")

    if ref_audio is None:
        st.warning("⚠️ Please upload a reference speaker audio file first!")
    
    input_type = st.radio("Choose input type:", ["Text", "Audio"])
    user_query = ""
    uploaded_voice = None

    if input_type == "Text":
        user_query = st.text_area("Enter your query:", "Hello, explain AI briefly.")
    elif input_type == "Audio":
        uploaded_voice = st.file_uploader("Upload an audio query (WAV format)", type=["wav", "ogg", "mp3"])
        
        if uploaded_voice is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(uploaded_voice.read())
                audio_path = temp_audio.name
            user_query = transcribe_audio(audio_path)
            os.unlink(audio_path)
            st.text_area("Transcribed Query:", user_query, disabled=False)
    
    if st.button("Generate Response", disabled=ref_audio is None or not user_query.strip()):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_speaker:
            temp_speaker.write(ref_audio.read())
            speaker_wav_path = temp_speaker.name
        
        response = get_llm_response(user_query)
        output_audio_path = "cloned_chat_response.wav"
        generate_speech(response, output_audio_path, speaker_wav_path, language_code)
        
        st.audio(output_audio_path, format="audio/wav")
        with open(output_audio_path, "rb") as f:
            st.download_button("Download Response Audio", f, file_name="cloned_chat_response.wav", mime="audio/wav")
        os.unlink(speaker_wav_path)
