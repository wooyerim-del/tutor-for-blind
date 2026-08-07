import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="Dual Hand AI Alphabet Tutor", layout="centered")

st.title("Dual Hand AI Alphabet Tutor")
st.write("Take a photo with your hands in front of the camera to detect letters.")

# MediaPipe Hand model setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmark IDs (Thumb, Index, Middle, Ring, Pinky)
finger_tips = [4, 8, 12, 16, 20]

# Letter mapping by finger count
right_letters = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
left_letters = {1: "F", 2: "G", 3: "H", 4: "I", 5: "J"}

# Web camera input
img_file_buffer = st.camera_input("Take a photo to detect alphabet")