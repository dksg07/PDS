import cv2
import mediapipe as mp
import torch
import numpy as np
import streamlit as st
from gtts import gTTS
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load a pre-trained model for sign language recognition (Placeholder)
class SignLanguageModel(torch.nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=21*3, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(128, 10)  # Example: 10 sign classes
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

model = SignLanguageModel()
model.load_state_dict(torch.load('sign_model.pth'))
model.eval()

# Function to extract hand landmarks
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return np.array(keypoints).reshape(1, -1)
    return None

# Real-time webcam processing
def process_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints(frame)
        if keypoints is not None:
            with torch.no_grad():
                keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
                prediction = model(keypoints_tensor)
                predicted_sign = torch.argmax(prediction, dim=1).item()
                cv2.putText(frame, f'Sign: {predicted_sign}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Sign Language Interpreter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Speech-to-Sign using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("start output.mp3")  # Change for Linux/Mac

# Streamlit UI
def main():
    st.title("AI-Powered Sign Language Interpreter")
    if st.button("Start Webcam"):
        process_webcam()
    user_text = st.text_input("Enter text to convert to speech:")
    if st.button("Convert to Speech"):
        text_to_speech(user_text)

if __name__ == "__main__":
    main()
