import whisper
from googletrans import Translator
from speechbrain.pretrained import SpeakerRecognition
import torch
import librosa
import numpy as np
import edge_tts  # Microsoft Edge-TTS for better voice control
import asyncio

def transcribe_audio(input_file):
    """Transcribes audio using OpenAI Whisper."""
    model = whisper.load_model("large")  # Use "large" for better accuracy
    result = model.transcribe(input_file)
    return result["text"]

def detect_gender(audio_path):
    """Detects speaker gender using SpeechBrain's SpeakerRecognition model."""
    recognizer = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="tmp_model"
    )
    
    # Load and compute speaker embeddings
    emb = recognizer.encode_batch([audio_path])
    
    # Estimate gender based on pitch as a fallback
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    return "male" if avg_pitch < 160 else "female"

async def text_to_speech(text, output_file, gender="male", lang="en"):
    """Converts text to speech using Edge-TTS with gender selection."""
    voice = "en-US-GuyNeural" if gender == "male" else "en-US-JennyNeural"
    tts = edge_tts.Communicate(text, voice)
    await tts.save(output_file)

def translate_text(text, target_language="es"):
    """Translates text to the target language."""
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

def process_audio(input_file, target_language="es", output_audio="dubbed_audio.mp3"):
    print("🔹 Transcribing audio...")
    transcribed_text = transcribe_audio(input_file)
    print("✅ Original Text:", transcribed_text)

    print("🔹 Detecting speaker gender...")
    speaker_gender = detect_gender(input_file)
    print(f"✅ Detected Gender: {speaker_gender}")

    print(f"🔹 Translating to {target_language}...")
    translated_text = translate_text(transcribed_text, target_language)
    print("✅ Translated Text:", translated_text)

    print("🔹 Generating gender-matched dubbed audio...")
    asyncio.run(text_to_speech(translated_text, output_audio, gender=speaker_gender, lang=target_language))
    print(f"✅ Dubbed audio saved as: {output_audio}")

# Example Usage
input_audio = "srk.wav"  # Replace with actual file
output_audio = "dubbed_audio.mp3"
target_lang = "en"

process_audio(input_audio, target_lang, output_audio)