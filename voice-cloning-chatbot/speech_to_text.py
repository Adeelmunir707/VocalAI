import whisper

def transcribe_audio(audio_path: str, model_size: str = "medium") -> str:
    """
    Transcribes an audio file using OpenAI's Whisper model.

    Parameters:
        audio_path (str): Path to the audio file.
        model_size (str): Whisper model size (default: "medium").

    Returns:
        str: Transcribed text from the audio file.
    """
    model = whisper.load_model(model_size)  # Load the specified Whisper model
    result = model.transcribe(audio_path)  # Transcribe the audio file
    return result["text"]  # Return the transcription

# Example usage
# if __name__ == "__main__":
#     audio_file = "Presentation 2voices/cloned_output_1.25x.wav"
#     transcription = transcribe_audio(audio_file)
#     print("Transcription:\n", transcription)
