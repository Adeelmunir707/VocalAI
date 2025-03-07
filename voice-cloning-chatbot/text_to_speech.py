from TTS.api import TTS
from pydub import AudioSegment

def text_to_speech(text: str, speaker_wav: str, output_file: str = "output.wav", speed: float = 1.25):
    """
    Converts text to speech using Coqui TTS with a reference speaker voice and speeds up the output.

    Parameters:
        text (str): The text to convert to speech.
        speaker_wav (str): Path to a reference audio file for voice cloning.
        output_file (str): The output audio file name (default: 'output.wav').
        speed (float): The speed factor to apply (default: 1.25x).
    """
    # Load the XTTS v2 model
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

    # Generate speech and save to file
    temp_output = "text_to_speech.wav"  # Temporary file before applying speed
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=temp_output)

    # Load and speed up the audio using pydub
    audio = AudioSegment.from_wav(temp_output)
    faster_audio = audio.speedup(playback_speed=speed)
    faster_audio.export(output_file, format="wav")

    print(f"Speech saved to {output_file} with {speed}x speed.")

# Example usage
if __name__ == "__main__":
    text = "Hello, this is a voice cloning demo."
    speaker_wav = "cloned_chat_response.wav"  # Replace with a real WAV file
    text_to_speech(text, speaker_wav, "cloned_output.wav", speed=1.25)
