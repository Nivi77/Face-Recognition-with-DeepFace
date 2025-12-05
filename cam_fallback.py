import cv2
from deepface import DeepFace
import pandas as pd

# ------------------ PARAMETERS ---------------------
DB_PATH = r"knownfaces/"

PRIMARY_MODEL = "Facenet512"
PRIMARY_THRESHOLD = 0.9

FALLBACK_MODEL = "SFace"
FALLBACK_THRESHOLD = 0.5   # SFace distances are lower

DETECTOR = "retinaface"

# ------------------ HELPER FUNCTION ---------------------
def verify_with_model(frame, model_name, threshold):
    try:
        result = DeepFace.find(
            img_path=frame,
            db_path=DB_PATH,
            model_name=model_name,
            detector_backend=DETECTOR,
            enforce_detection=False,
            distance_metric="euclidean_l2"
        )

        if isinstance(result, list) and len(result) > 0 and not result[0].empty:
            row = result[0].iloc[0]
            dist = row["distance"]
            identity = row["identity"]

            if dist < threshold:
                return identity, dist

        return None, None

    except Exception as e:
        print("Error:", e)
        return None, None


# ------------------ WEBCAM LOOP ---------------------
cap = cv2.VideoCapture(0)

print("Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for DeepFace
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- 1️⃣ Try PRIMARY model (Facenet512) ----
    identity, dist = verify_with_model(rgb_frame, PRIMARY_MODEL, PRIMARY_THRESHOLD)

    # ---- 2️⃣ If primary fails, try fallback (SFace) ----
    if identity is None:
        identity, dist = verify_with_model(rgb_frame, FALLBACK_MODEL, FALLBACK_THRESHOLD)

    # ---- Display result ----
    if identity:
        cv2.putText(frame, f"MATCH: {identity.split('/')[-1]}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Distance: {dist:.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    else:
        cv2.putText(frame, "NO MATCH", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Dual-Model Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
