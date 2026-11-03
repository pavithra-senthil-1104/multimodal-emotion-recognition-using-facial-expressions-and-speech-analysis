
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import threading

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import GlorotUniform

from src.realtime import predict_speech
from src.fusion import combine_emotions


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "face_model.keras"
TEMP_DIR = ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 22050
RECORD_SECONDS = 5
IMAGE_SIZE = (160, 160)

EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

face_image_path = TEMP_DIR / "captured_face.jpg"
audio_path = TEMP_DIR / "recorded_audio.wav"

captured_face = False
recorded_audio = False
busy = False


# =========================================================
# KERAS COMPATIBILITY
# =========================================================

class CompatibleGlorotUniform(GlorotUniform):
    """Handle extra initializer settings from some Keras versions."""

    def __init__(
        self,
        seed=None,
        input_axes=None,
        output_axes=None,
        **kwargs
    ):
        super().__init__(seed=seed, **kwargs)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("input_axes", None)
        config.pop("output_axes", None)
        return cls(**config)


# =========================================================
# LOAD FACIAL MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Facial model not found:\n{MODEL_PATH}\n\n"
        "Train the model first using:\n"
        "python src/train_face.py"
    )

try:
    face_model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "GlorotUniform": CompatibleGlorotUniform,
            "keras>GlorotUniform": CompatibleGlorotUniform,
        },
    )

    print("Facial model loaded successfully!")

except Exception as exc:
    raise RuntimeError(
        "Could not load the facial emotion model.\n\n"
        "The saved model may be incompatible with the installed "
        "TensorFlow/Keras version.\n\n"
        "Check your versions and, if necessary, retrain the facial "
        "model using the current Python environment.\n\n"
        f"Original error:\n{exc}"
    ) from exc


# =========================================================
# CAPTURE FACE FROM CAMERA
# =========================================================

def capture_face():
    global captured_face

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        messagebox.showerror(
            "Camera Error",
            "Could not open your laptop camera.\n"
            "Close other applications using the camera and try again."
        )
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    selected_face = None
    window_name = "Capture Face - SPACE to capture, ESC to cancel"

    cv2.namedWindow(window_name)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                messagebox.showerror(
                    "Camera Error",
                    "Could not read a camera frame."
                )
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            if len(faces) > 0:
                x, y, w, h = max(
                    faces,
                    key=lambda box: box[2] * box[3]
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "Face detected - press SPACE",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # ESC cancels capture
            if key == 27:
                break

            # SPACE captures the detected face
            if key == 32 and len(faces) > 0:
                x, y, w, h = max(
                    faces,
                    key=lambda box: box[2] * box[3]
                )

                margin = int(0.15 * max(w, h))
                height, width = frame.shape[:2]

                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(width, x + w + margin)
                y2 = min(height, y + h + margin)

                selected_face = frame[y1:y2, x1:x2].copy()
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if selected_face is not None:
        saved = cv2.imwrite(
            str(face_image_path),
            selected_face
        )

        if not saved:
            messagebox.showerror(
                "Save Error",
                "Could not save the captured face."
            )
            return

        captured_face = True

        face_status.config(
            text="Face captured ✓",
            fg="green"
        )

        show_face_preview(selected_face)


# =========================================================
# DISPLAY CAPTURED FACE
# =========================================================

def show_face_preview(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)
    image.thumbnail((220, 160))

    photo = ImageTk.PhotoImage(image)

    face_preview.config(
        image=photo,
        text=""
    )

    # Keep a reference so Tkinter does not remove the image
    face_preview.image = photo


# =========================================================
# RECORD AUDIO FROM MICROPHONE
# =========================================================

def record_audio():
    global recorded_audio

    record_button.config(state="disabled")

    try:
        audio_status.config(
            text="Recording... speak now!",
            fg="red"
        )

        root.update_idletasks()

        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )

        sd.wait()

        if (
            audio.size == 0
            or float(np.sqrt(np.mean(audio ** 2))) < 1e-5
        ):
            recorded_audio = False

            audio_status.config(
                text="Recording too quiet. Try again.",
                fg="red",
            )
            return

        sf.write(
            str(audio_path),
            audio,
            SAMPLE_RATE,
            subtype="PCM_16",
        )

        recorded_audio = True

        audio_status.config(
            text=f"Audio recorded ✓ ({RECORD_SECONDS} seconds)",
            fg="green",
        )

    except Exception as exc:
        recorded_audio = False

        messagebox.showerror(
            "Microphone Error",
            str(exc)
        )

    finally:
        record_button.config(state="normal")


# =========================================================
# PREDICT FACIAL EMOTION
# =========================================================

def predict_face(image_path):
    image = (
        Image.open(image_path)
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing: pixel values become [-1, 1]
    array = tf.keras.applications.mobilenet_v2.preprocess_input(
        array
    )

    array = np.expand_dims(array, axis=0)

    probabilities = face_model.predict(
        array,
        verbose=0
    )[0]

    if len(probabilities) != len(EMOTIONS):
        raise ValueError(
            f"Face model returned {len(probabilities)} classes, "
            f"but {len(EMOTIONS)} emotion labels are configured."
        )

    index = int(np.argmax(probabilities))

    emotion = EMOTIONS[index]
    confidence = float(probabilities[index])

    return emotion, confidence


# =========================================================
# ANALYZE FACE + SPEECH + FUSION
# =========================================================

def analyze_emotions():
    global busy

    if busy:
        return

    if not captured_face:
        messagebox.showwarning(
            "Missing Face",
            "Please capture your face first."
        )
        return

    if not recorded_audio:
        messagebox.showwarning(
            "Missing Audio",
            "Please record your voice first."
        )
        return

    busy = True

    analyze_button.config(state="disabled")
    result_label.config(text="Analyzing... please wait")

    def work():
        try:
            # Facial prediction
            face_emotion, confidence = predict_face(
                face_image_path
            )

            # Speech prediction
            speech_emotion = predict_speech(
                audio_path
            )

            # Combine the two predictions
            result = combine_emotions(
                face_emotion,
                speech_emotion
            )

            root.after(
                0,
                lambda fe=face_emotion,
                       c=confidence,
                       se=speech_emotion,
                       r=result: show_results(fe, c, se, r)
            )

        except Exception as exc:
            error_text = str(exc)

            root.after(
                0,
                lambda err=error_text: messagebox.showerror(
                    "Analysis Error",
                    err
                )
            )

            root.after(
                0,
                lambda: result_label.config(
                    text="Analysis failed."
                )
            )

        finally:
            root.after(0, finish_analysis)

    threading.Thread(
        target=work,
        daemon=True
    ).start()


# =========================================================
# DISPLAY PREDICTION RESULTS
# =========================================================

def show_results(
    face_emotion,
    confidence,
    speech_emotion,
    result
):
    face_result.config(
        text=(
            f"Facial emotion: {face_emotion.title()}\n"
            f"Model score: {confidence:.1%}"
        )
    )

    speech_result.config(
        text=f"Speech emotion: {speech_emotion.title()}"
    )

    if result["agreement"]:
        result_label.config(
            text=(
                f"Combined result: "
                f"{result['final_emotion'].title()}\n"
                "Both models agree."
            )
        )
    else:
        result_label.config(
            text=(
                f"Combined result: "
                f"{result['final_emotion'].title()}\n"
                "Both models agree."
            )
        )


# =========================================================
# ENABLE BUTTON AFTER ANALYSIS
# =========================================================

def finish_analysis():
    global busy

    busy = False
    analyze_button.config(state="normal")


# =========================================================
# CLEAR / RESET INPUTS AND RESULTS
# =========================================================

def clear_inputs():
    global captured_face, recorded_audio

    captured_face = False
    recorded_audio = False

    face_status.config(
        text="No face captured",
        fg="gray"
    )

    audio_status.config(
        text="No audio recorded",
        fg="gray"
    )

    face_preview.config(
        image="",
        text="Camera preview"
    )

    face_preview.image = None

    face_result.config(
        text="Facial emotion: --"
    )

    speech_result.config(
        text="Speech emotion: --"
    )

    result_label.config(
        text="Combined result: --"
    )


# =========================================================
# BUILD TKINTER GUI
# =========================================================

root = tk.Tk()

root.title("Multimodal Emotion Recognition")
root.geometry("700x690")
root.minsize(600, 620)
root.configure(bg="#f2f5f9")


# Heading
title = tk.Label(
    root,
    text="Multimodal Emotion Recognition",
    font=("Arial", 20, "bold"),
    bg="#f2f5f9",
    fg="#1d3557",
)

title.pack(pady=(18, 5))


subtitle = tk.Label(
    root,
    text="Facial Expression + Speech Analysis",
    font=("Arial", 11),
    bg="#f2f5f9",
    fg="#52657a",
)

subtitle.pack(pady=(0, 15))


# Input section
input_frame = tk.Frame(
    root,
    bg="#f2f5f9"
)

input_frame.pack(
    fill="x",
    padx=20
)


# Camera section
face_frame = tk.LabelFrame(
    input_frame,
    text="1. Laptop Camera",
    font=("Arial", 11, "bold"),
    padx=12,
    pady=12,
    bg="white",
)

face_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 8)
)


face_preview = tk.Label(
    face_frame,
    text="Camera preview",
    width=24,
    height=7,
    bg="#e7edf5",
)

face_preview.pack(pady=8)


tk.Button(
    face_frame,
    text="Capture Face",
    command=capture_face,
    width=18,
).pack(pady=5)


face_status = tk.Label(
    face_frame,
    text="No face captured",
    bg="white",
    fg="gray",
)

face_status.pack()


# Microphone section
audio_frame = tk.LabelFrame(
    input_frame,
    text="2. Laptop Microphone",
    font=("Arial", 11, "bold"),
    padx=12,
    pady=12,
    bg="white",
)

audio_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(8, 0)
)


tk.Label(
    audio_frame,
    text=f"Record {RECORD_SECONDS} seconds\nSpeak naturally.",
    font=("Arial", 12),
    bg="white",
    justify="center",
).pack(pady=22)


record_button = tk.Button(
    audio_frame,
    text="Record Voice",
    command=record_audio,
    width=18,
)

record_button.pack(pady=12)


audio_status = tk.Label(
    audio_frame,
    text="No audio recorded",
    bg="white",
    fg="gray",
    wraplength=220,
)

audio_status.pack(pady=8)


# Analyze button
analyze_button = tk.Button(
    root,
    text="ANALYZE EMOTION",
    command=analyze_emotions,
    font=("Arial", 12, "bold"),
    bg="#2457a7",
    fg="white",
    activebackground="#17417f",
    activeforeground="white",
    padx=20,
    pady=10,
)

analyze_button.pack(pady=18)


# Results section
results_frame = tk.LabelFrame(
    root,
    text="Prediction Results",
    font=("Arial", 12, "bold"),
    padx=15,
    pady=12,
    bg="white",
)

results_frame.pack(
    fill="x",
    padx=20,
    pady=(0, 10)
)


face_result = tk.Label(
    results_frame,
    text="Facial emotion: --",
    font=("Arial", 11),
    bg="white",
)

face_result.pack(
    anchor="w",
    pady=4
)


speech_result = tk.Label(
    results_frame,
    text="Speech emotion: --",
    font=("Arial", 11),
    bg="white",
)

speech_result.pack(
    anchor="w",
    pady=4
)


result_label = tk.Label(
    results_frame,
    text="Combined result: --",
    font=("Arial", 12, "bold"),
    bg="white",
    fg="#2457a7",
    justify="left",
)

result_label.pack(
    anchor="w",
    pady=8
)


# Reset button
tk.Button(
    root,
    text="Clear / Reset",
    command=clear_inputs,
    width=18,
).pack(pady=(0, 12))


# =========================================================
# START APPLICATION
# =========================================================

root.mainloop()
