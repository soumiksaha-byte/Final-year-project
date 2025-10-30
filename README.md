🧠 Overall Project Workflow

our system has three main stages (three Python files):

Stage	File Name	Purpose
1️⃣ Data Collection	face_collect.py	Captures face images (full frames) for each student.
2️⃣ Model Training	train_model.py	Extracts embeddings from collected images and saves them.
3️⃣ Recognition & Attendance	face_recognition_attendance.py	Detects and recognizes faces in real-time, marks attendance in CSV.
⚙️ 1️⃣ collect_faces.py — Face Data Collection
Purpose:

Collects raw face data for each student.

Key Steps:

Ask for:

University ID

Full Name

Create folder → TrainingImage/<person_id>/

Open webcam → detect faces using InsightFace.

For each frame where a face is detected:

Draw a rectangle around the face.

Save the entire frame as an image (not cropped).

Continue until 50 images are saved.

Output:

📁 TrainingImage/<person_id>/*.jpg


✅ Result: You now have image data for each person stored by ID.

🧩 2️⃣ train_model.py — Model Training with InsightFace
Purpose:

Generate embeddings (numerical face features) for all collected images and store them.

Process:

Read each folder in TrainingImage/.

For each image:

Detect the face.

Generate its face embedding using FaceAnalysis.

Save all embeddings in:

TrainingImageLabel/insightface_embeddings.pkl


Structure inside the pickle file:

{
    "ids": [person_id1, person_id2, ...],
    "names": [name1, name2, ...],
    "embeddings": [embedding1, embedding2, ...]
}


✅ Result: our model now “knows” each person’s facial features.

🧾 3️⃣ attendance.py — Real-Time Recognition
Purpose:

Recognize faces live from the webcam and mark attendance automatically.

Workflow:

Load embeddings from insightface_embeddings.pkl

Start webcam.

For every detected face:

Extract its embedding.

Compare with all stored embeddings using cosine similarity.

Find the most similar match.

If similarity > threshold (e.g. 0.4):

Display name + ID above the bounding box (green).

Mark attendance in Attendance.csv with timestamp.

If not matched:

Label as Unknown (red box).
📁 FaceRecognitionSystem/
│
├── collect_faces.py                 # Collects face data
├── train_model.py                  # Generates embeddings
├── attendance.py  # Real-time attendance
│
├── 📁 TrainingImage/
│   ├── 1234/
│   │   ├── 1234_1.jpg
│   │   ├── 1234_2.jpg
│   │   └── ...
│   └── 5678/
│       ├── 5678_1.jpg
│       └── ...
│
├── 📁 TrainingImageLabel/
│   └── insightface_embeddings.pkl
│
└── Attendance.csv
