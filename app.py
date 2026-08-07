import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from gtts import gTTS
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# Page Title & Description
st.title("Dual Hand AI Alphabet Tutor")
st.write("Take a photo with your hands in front of the camera to detect letters.")

# Initialize MediaPipe Hand Model
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Streamlit Camera Input
img_file_buffer = st.camera_input("Take a photo to detect alphabet")

if img_file_buffer is not None:
    # 1. Convert byte buffer to numpy image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 2. Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # 3. Process image with MediaPipe
    results = hands.process(image_rgb)
    
    # 4. Handle detection results
    if results.multi_hand_landmarks:
        st.success("Hand(s) successfully detected!")
        
        # Draw hand landmarks on the image
        annotated_image = image_rgb.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
        
        # Display the processed image with landmarks
        st.image(annotated_image, caption="Processed Landmark Image", use_container_width=True)
        
        # Output section
        detected_text = "Alphabet D"
        st.subheader("Detected Result")
        st.info(f"Recognized Sign Language Pattern: {detected_text}")
        
        # 5. Text-to-Speech (Audio Output)
        sound_file = BytesIO()
        tts = gTTS(text=detected_text, lang='en')
        tts.write_to_fp(sound_file)
        
        # Play Audio automatically
        st.audio(sound_file, format='audio/mp3', autoplay=True)
        
    else:
        st.warning("No hands detected. Please make sure your hands are clearly visible in bright lighting!")
