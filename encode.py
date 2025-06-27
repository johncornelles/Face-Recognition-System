# encode_faces.py
import face_recognition
import os
import pickle
import cv2

# Directory with known faces
KNOWN_FACES_DIR = "known_faces"
encodeList = []
names = []

print("[INFO] Encoding faces...")

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encodeList.append(encodings[0])
            names.append(os.path.splitext(filename)[0])
            print(f"[ENCODED] {filename}")
        else:
            print(f"[SKIPPED] {filename} - No face found")

# Save encodings
with open("EncodeFile.p", "wb") as f:
    pickle.dump((encodeList, names), f)

print("[DONE] Encodings saved to EncodeFile.p")
