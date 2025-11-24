import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("clear_data_sets/vsd (2).mp4")  # Replace with your video file

all_landmarks = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
        else:
            # If no pose detected, fill with zeros
            landmarks = np.zeros((33, 4))

        all_landmarks.append(landmarks)

        cv2.imshow('Analyzing Video', image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit early
            break

cap.release()
cv2.destroyAllWindows()

# Save all landmarks as a numpy array: shape (num_frames, 33, 4)
all_landmarks = np.array(all_landmarks)
os.makedirs('processed_data', exist_ok=True)
np.save('processed_data/ref_landmarks.npy', all_landmarks)
print(f"Saved landmarks for {len(all_landmarks)} frames to processed_data/ref_landmarks.npy")