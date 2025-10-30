import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

dataset_dir = "TrainingImage"
embeddings = []
names = []

print("🧠 Training model...")

for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        # Convert to RGB for InsightFace
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)

        if len(faces) > 0:
            emb = faces[0].embedding
            embeddings.append(emb)
            names.append(person_name)
        else:
            print(f"⚠️ No face found in: {img_path}")

# Save embeddings
os.makedirs("TrainingImageLabel", exist_ok=True)
with open("TrainingImageLabel/insightface_embeddings.pkl", "wb") as f:
    pickle.dump({"embeddings": embeddings, "names": names}, f)

print(f"✅ Training complete! Total faces trained: {len(embeddings)}")
