from deepface import DeepFace
import pandas as pd

# Primary and fallback models
PRIMARY_MODEL = "Facenet512"
FALLBACK_MODEL = "SFace"

# Per-model thresholds (very important)
THRESHOLDS = {
    "Facenet512": 0.90,
    "SFace": 0.75
}

IMG = r"test_images\Screenshot 2025-12-05 191926.png"
DB = r"knownfaces/"

def try_model(model_name):
    """
    Attempts matching using the given model.
    Returns (match_name, distance) or (None, None)
    """
    result = DeepFace.find(
        img_path=IMG,
        db_path=DB,
        model_name=model_name,
        detector_backend="retinaface",
        enforce_detection=True,
        distance_metric="euclidean_l2",
        silent=True
    )

    if isinstance(result, list) and len(result) > 0 and not result[0].empty:
        row = result[0].iloc[0]
        distance = row["distance"]
        identity = row["identity"]

        if distance < THRESHOLDS[model_name]:
            name = identity.split("/")[-1].split(".")[0]
            return (name, distance)

    return (None, None)


# ---------------- MAIN EXECUTION ------------------

# 1️⃣ Try Facenet512
match, dist = try_model(PRIMARY_MODEL)

if match:
    print(f"Match found (Primary: {PRIMARY_MODEL}): {match} | Distance: {dist}")
else:
    print("Primary model failed. Trying fallback model...")

    # 2️⃣ Try SFace
    match, dist = try_model(FALLBACK_MODEL)

    if match:
        print(f"Match found (Fallback: {FALLBACK_MODEL}): {match} | Distance: {dist}")
    else:
        print("❌ No match found with both models.")
