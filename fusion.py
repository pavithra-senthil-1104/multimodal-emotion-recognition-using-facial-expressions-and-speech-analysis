
def combine_emotions(
    face_emotion,
    speech_emotion,
    face_weight=0.5,
    speech_weight=0.5
):
    face_emotion = face_emotion.lower()
    speech_emotion = speech_emotion.lower()

    if face_emotion == speech_emotion:
        final_emotion = face_emotion
    elif face_weight > speech_weight:
        final_emotion = face_emotion
    elif speech_weight > face_weight:
        final_emotion = speech_emotion
    else:
        final_emotion = "mixed"

    return {
        "face_emotion": face_emotion,
        "speech_emotion": speech_emotion,
        "final_emotion": final_emotion,
        "agreement": face_emotion == speech_emotion
    }


if __name__ == "__main__":
    result = combine_emotions("happy", "sad")
    print(result)
