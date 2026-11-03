w3from pathlib import Path
import joblib
import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "speech_svm.joblib"


def extract_features(file_path):
    audio, sr = librosa.load(
        str(file_path), sr=22050, mono=True
    )

    if len(audio) == 0:
        raise ValueError("Empty audio file")

    if np.sqrt(np.mean(audio ** 2)) < 1e-5:
        raise ValueError("Audio is silent")

    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=40,
        n_fft=2048, hop_length=512
    )
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr
    )

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        [
            np.mean(zcr), np.std(zcr),
            np.mean(centroid), np.std(centroid)
        ]
    ]).astype(np.float32)


def predict_speech(audio_path):
    saved = joblib.load(MODEL_PATH)
    features = extract_features(audio_path)
    prediction = saved["model"].predict([features])[0]
    return str(prediction)


if __name__ == "__main__":
    path = input("Enter WAV file path: ").strip().strip('"')
    print("Speech emotion:", predict_speech(path))
