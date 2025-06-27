# main.py
import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime

# Load encodings
with open("EncodeFile.p", "rb") as f:
    encodeListKnown, knownNames = pickle.load(f)

# Local in-memory database (simulate Firebase)
student_db = {
    "Musk": {
        "name": "Elon Musk",
        "major": "Physics",
        "starting_year": 2020,
        "total_attendance": 0,
        "last_attendance_time": "2000-01-01 00:00:00"
    },
    "Me": {
        "name": "John",
        "major": "AI & Trading",
        "starting_year": 2023,
        "total_attendance": 0,
        "last_attendance_time": "2000-01-01 00:00:00"
    }
}

print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize and convert
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    faces = face_recognition.face_locations(rgb_small_frame)
    encodings = face_recognition.face_encodings(rgb_small_frame, faces)

    for encodeFace, faceLoc in zip(encodings, faces):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = knownNames[matchIndex]
            student = student_db.get(name, None)
            if student:
                last_attendance = datetime.strptime(student["last_attendance_time"], "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - last_attendance).total_seconds() > 30:
                    student["total_attendance"] += 1
                    student["last_attendance_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[ATTENDANCE] {student['name']} - Total: {student['total_attendance']}")

                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, student["name"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
