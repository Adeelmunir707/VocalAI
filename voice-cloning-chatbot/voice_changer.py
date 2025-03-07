import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from scipy.signal import lfilter

def change_voice(input_file, output_file, mode="deep"):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)

    if mode == "deep":
        # Lower pitch by 4 semitones (natural deep voice)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)

        # Apply noise reduction to keep sound clean
        y_denoised = nr.reduce_noise(y=y_shifted, sr=sr)

        # Apply a soft low-pass filter for smoothness
        b, a = [1], [1, -0.7]  # Soft low-pass filter
        y_final = lfilter(b, a, y_denoised)

    elif mode == "robotic":
        # Increase pitch by 3 semitones
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)

        # Apply a tremolo effect (modulated volume for robotic sound)
        mod_freq = 5  # Hz
        t = np.linspace(0, len(y_shifted) / sr, num=len(y_shifted))
        tremolo = 0.8 + 0.2 * np.sin(2 * np.pi * mod_freq * t)
        y_final = y_shifted * tremolo

    # Save the processed file
    sf.write(output_file, y_final, sr)
    print(f"Processed file saved as: {output_file}")

# Example usage
input_voice = "srk.wav"  # Replace with your file
output_voice1 = "voice_changer1.wav"
output_voice2 = "voice_changer.wav"

change_voice(input_voice, output_voice1, mode="deep")      # Deep Voice Effect (Cleaner)
change_voice(input_voice, output_voice2, mode="robotic")  # Robotic Voice Effect
