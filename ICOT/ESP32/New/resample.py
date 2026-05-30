import librosa
import soundfile as sf

TARGET_SR = 16000

def resample_audio(input_path, output_path, target_sr=TARGET_SR):
    # Load audio (mono + resample exactly like librosa.load)
    audio, _ = librosa.load(input_path, sr=target_sr, mono=True)

    # Save resampled audio
    sf.write(output_path, audio, target_sr, subtype="PCM_16")

    print(f"Saved resampled audio to: {output_path}")


# Example usage
# resample_audio("/home/sachcith/Downloads/VD001_Active.wav", "output.wav")
resample_audio("/home/sachcith/Documents/Sem 4/IOT/Project/MissingQueen.wav", "output.wav")
