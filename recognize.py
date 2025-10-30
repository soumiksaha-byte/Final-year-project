import cv2
import numpy as np
import pickle
from datetime import datetime
import csv
import os
from insightface.app import FaceAnalysis

# ------------------------------
# Load trained embeddings
# ------------------------------
with open("TrainingImageLabel/insightface_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
embeddings = np.array(data["embeddings"])
names = np.array(data["names"])

# ------------------------------
# Initialize InsightFace model
# ------------------------------
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# ------------------------------
# Attendance Setup
# ------------------------------
attendance_file = "Attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

marked_today = set()

def mark_attendance(name):
    """Mark attendance if not already marked today"""
    today = datetime.now().strftime("%Y-%m-%d")
    key = (name, today)
    if key not in marked_today:
        now = datetime.now()
        with open(attendance_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
        marked_today.add(key)
        print(f"✅ Attendance marked for {name}")

# ------------------------------
# Start webcam
# ------------------------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam.open(0)

print("🎥 Press 'q' to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    faces = app.get(frame)
    for face in faces:
        emb = face.embedding

        # Compute cosine similarity
        sims = np.dot(embeddings, emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
        )
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        # Recognition threshold
        if best_score > 0.38:
            name = names[best_idx]
            color = (0, 255, 0)
            mark_attendance(name)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Draw bounding box and label
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("InsightFace Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("🛑 Program ended.")
