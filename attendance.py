import cv2
import numpy as np
import pickle
from datetime import datetime
from insightface.app import FaceAnalysis

# Load embeddings
with open("TrainingImageLabel/insightface_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = np.array(data["embeddings"])
names = np.array(data["names"])

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Attendance setup
attendance_file = "Attendance.csv"
try:
    open(attendance_file, 'x').close()
except FileExistsError:
    pass

def mark_attendance(name):
    with open(attendance_file, 'r+', newline='') as f:
        existing_data = f.readlines()
        names_list = [line.split(',')[0] for line in existing_data]
        if name not in names_list:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{name},{now}\n")
            print(f"✅ Attendance marked for {name}")

# Start webcam
cam = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        emb = face.embedding
        similarities = np.dot(embeddings, emb) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
        )
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # Threshold tuning
        if best_score > 0.4:
            name = names[best_idx]
            color = (0, 255, 0)
            mark_attendance(name)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("InsightFace Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()



