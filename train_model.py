import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# -------------------------------
# Take user input for folder name
# -------------------------------
person_id = input("Enter the name to train (e.g., 12345): ").strip()
dataset_dir = os.path.join("TrainingImage", person_id)

if not os.path.exists(dataset_dir):
    print(f"❌ Folder not found: {dataset_dir}")
    exit()

# -------------------------------
# Initialize InsightFace
# -------------------------------
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ InsightFace model initialized.")
print(f"🧠 Training on folder: {dataset_dir}")

embeddings = []
names = []

# -------------------------------
# Process each image in the folder
# -------------------------------
for img_name in os.listdir(dataset_dir):
    img_path = os.path.join(dataset_dir, img_name)
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
        names.append(person_id)
    else:
        print(f"⚠️ No face detected in {img_path}")

# -------------------------------
# Save embeddings
# -------------------------------
os.makedirs("TrainingImageLabel", exist_ok=True)
embed_file = "TrainingImageLabel/insightface_embeddings.pkl"

# Load existing data (if available)
if os.path.exists(embed_file):
    with open(embed_file, "rb") as f:
        data = pickle.load(f)
    embeddings = np.vstack([data["embeddings"], embeddings]) if len(data["embeddings"]) else np.array(embeddings)
    names = np.concatenate([data["names"], names])
else:
    embeddings = np.array(embeddings)
    names = np.array(names)

# Save updated embeddings
with open(embed_file, "wb") as f:
    pickle.dump({"embeddings": embeddings, "names": names}, f)

print(f"✅ Training complete for {person_id}!")
