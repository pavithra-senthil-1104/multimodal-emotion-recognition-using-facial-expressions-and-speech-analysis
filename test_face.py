
import numpy as np
import tensorflow as tf

from pathlib import Path
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "face_images"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "face_model.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 40
SEED = 42

EMOTION_LABELS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

tf.keras.utils.set_random_seed(SEED)


# ==========================================
# 2. Load image paths and labels
# ==========================================

def load_dataset():
    image_paths = []
    labels = []
    groups = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Face image folder not found: {DATA_DIR}"
        )

    for emotion in EMOTION_LABELS:
        emotion_dir = DATA_DIR / emotion

        if not emotion_dir.exists():
            print(f"Warning: missing folder: {emotion_dir}")
            continue

        for path in emotion_dir.iterdir():
            if path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png"
            }:
                continue

            # Group frames from the same original video together.
            # This helps prevent train/test data leakage.
            group = path.stem.split("_frame_")[0]

            image_paths.append(str(path))
            labels.append(emotion)
            groups.append(group)

    if not image_paths:
        raise ValueError(
            f"No images found in {DATA_DIR}. "
            "Check your extracted JPG files."
        )

    return (
        np.array(image_paths),
        np.array(labels),
        np.array(groups),
    )


# ==========================================
# 3. Read and preprocess images
# ==========================================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # MobileNetV2 expects pixel values scaled to [-1, 1].
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


# ==========================================
# 4. Build TensorFlow datasets
# ==========================================

def make_tf_dataset(paths, labels, training=False):
    label_to_id = {
        emotion: index
        for index, emotion in enumerate(EMOTION_LABELS)
    }

    numeric_labels = np.array([
        label_to_id[label] for label in labels
    ], dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, numeric_labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==========================================
# 5. Create MobileNetV2 model
# ==========================================

def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ], name="augmentation")

    base_model = (
        tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )
    )

    # Train the classification head first.
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="face_image"
    )

    x = augmentation(inputs)

    # Keep pretrained Batch Normalization layers in inference mode.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        len(EMOTION_LABELS),
        activation="softmax",
        name="emotion_probabilities"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================
# 6. Train and evaluate
# ==========================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading facial emotion images...")

    paths, labels, groups = load_dataset()

    print(f"Total images: {len(paths)}")
    print(f"Unique source videos: {len(np.unique(groups))}")
    print("Images per emotion:", dict(Counter(labels)))

    if len(np.unique(groups)) < 2:
        raise ValueError(
            "At least two source videos are needed "
            "for a train/test split."
        )

    # Split by original video, not by individual frames.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED
    )

    train_idx, test_idx = next(
        splitter.split(paths, labels, groups)
    )

    train_paths = paths[train_idx]
    train_labels = labels[train_idx]

    test_paths = paths[test_idx]
    test_labels = labels[test_idx]

    print(f"\nTraining images: {len(train_paths)}")
    print(f"Testing images: {len(test_paths)}")

    train_ds = make_tf_dataset(
        train_paths, train_labels, training=True
    )

    test_ds = make_tf_dataset(
        test_paths, test_labels
    )

    print("\nBuilding MobileNetV2 model...")
    model = build_model()

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    print("\nStarting training...")

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Load the best checkpoint saved during training.
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nEvaluating model...")
    probabilities = model.predict(test_ds)

    predicted_ids = np.argmax(probabilities, axis=1)

    label_to_id = {
        emotion: index
        for index, emotion in enumerate(EMOTION_LABELS)
    }

    actual_ids = np.array([
        label_to_id[label] for label in test_labels
    ])

    print("\nClassification report:")
    print(classification_report(
        actual_ids,
        predicted_ids,
        labels=list(range(len(EMOTION_LABELS))),
        target_names=EMOTION_LABELS,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        actual_ids,
        predicted_ids,
        labels=list(range(len(EMOTION_LABELS)))
    ))

    print(f"\nModel saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
