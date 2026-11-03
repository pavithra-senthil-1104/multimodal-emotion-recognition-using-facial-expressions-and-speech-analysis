
from pathlib import Path

import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================
# 1. PATHS AND SETTINGS
# =====================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "audio"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050

EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]


# =====================================
# 2. EXTRACT MFCC FEATURES
# =====================================

def extract_features(file_path):

    audio, sr = librosa.load(
        str(file_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Empty audio file")

    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < 1e-5:
        raise ValueError(
            f"Audio is silent or too quiet (RMS={rms:.2e})"
        )

    # Normalize waveform
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )

    # Changes in MFCC features over time
    delta = librosa.feature.delta(mfcc)

    # Additional acoustic features
    zcr = librosa.feature.zero_crossing_rate(y=audio)

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    # Mean and standard deviation features
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        [
            np.mean(zcr),
            np.std(zcr),
            np.mean(centroid),
            np.std(centroid)
        ]
    ])

    return features.astype(np.float32)


# =====================================
# 3. LOAD DATASET
# =====================================

def load_dataset():

    X = []
    y = []
    skipped = 0

    for emotion in EMOTIONS:

        folder = DATA_DIR / emotion

        if not folder.exists():
            print(f"WARNING: Missing folder: {folder}")
            continue

        files = sorted(folder.glob("*.wav"))

        print(f"{emotion}: {len(files)} files")

        for file_path in files:

            try:
                features = extract_features(file_path)

                X.append(features)
                y.append(emotion)

            except Exception as exc:
                skipped += 1
                print(f"Skipped {file_path.name}: {exc}")

    if len(X) == 0:
        raise RuntimeError(
            "No usable audio files found. "
            "Check dataset/audio and the emotion folders."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    print("\n========== DATASET SUMMARY ==========")
    print("Usable files:", len(y))
    print("Skipped files:", skipped)
    print("Feature shape:", X.shape)

    for emotion in EMOTIONS:
        print(f"{emotion}: {int(np.sum(y == emotion))}")

    return X, y


# =====================================
# 4. TRAIN AND TUNE SVM
# =====================================

def main():

    X, y = load_dataset()

    unique_classes, counts = np.unique(
        y,
        return_counts=True
    )

    if len(unique_classes) < 2 or np.min(counts) < 2:
        raise RuntimeError(
            "Each emotion class needs at least two usable files."
        )

    # Hold out test data until final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Ensure enough samples per class for five-fold CV
    train_classes, train_counts = np.unique(
        y_train,
        return_counts=True
    )

    if np.min(train_counts) < 2:
        raise RuntimeError(
            "Not enough training samples for stratified "
            "cross-validation."
        )

    n_splits = min(5, int(np.min(train_counts)))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    pipeline = make_pipeline(
        StandardScaler(),
        SVC(class_weight="balanced")
    )

    param_grid = [
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 3, 10, 30, 100],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1]
        },
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1, 3, 10, 30, 100]
        }
    ]

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    print("\n========== SVM HYPERPARAMETER TUNING ==========")
    print("Searching for the best settings...")

    search.fit(X_train, y_train)

    model = search.best_estimator_

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        "Best CV macro F1:",
        f"{search.best_score_ * 100:.2f}%"
    )

    # =====================================
    # 5. FINAL TEST EVALUATION
    # =====================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=EMOTIONS,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=EMOTIONS
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(cm)

    # Plot and save confusion matrix
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=EMOTIONS
    ).plot(xticks_rotation=45)

    plt.title("Speech Emotion Recognition - Tuned SVM")
    plt.tight_layout()

    plot_path = MODEL_DIR / "speech_confusion_matrix.png"

    plt.savefig(plot_path, dpi=150)
    print(f"\nConfusion matrix saved: {plot_path}")

    plt.show()

    # =====================================
    # 6. SAVE MODEL
    # =====================================

    model_path = MODEL_DIR / "speech_svm.joblib"

    joblib.dump(
        {
            "model": model,
            "emotions": EMOTIONS,
            "sample_rate": SAMPLE_RATE
        },
        model_path
    )

    print(f"\nModel saved: {model_path}")
    print("Speech emotion training completed!")


if __name__ == "__main__":
    main()
