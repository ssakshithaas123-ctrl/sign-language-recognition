import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import av
import time
import os

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real-Time Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Remove excessive top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #a7adb8;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Information cards */
    .info-card {
        background-color: #171b24;
        border: 1px solid #292f3a;
        border-radius: 12px;
        padding: 20px;
        min-height: 145px;
    }

    .info-card h3 {
        margin-top: 0;
        font-size: 18px;
    }

    .info-card p {
        color: #b7bdc8;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Camera permission box */
    .permission-box {
        background-color: #151a23;
        border: 1px solid #344052;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }

    .permission-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 7px;
    }

    .permission-text {
        color: #b8bec9;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Recognition box */
    .recognition-box {
        background-color: #171b24;
        border: 1px solid #292f3a;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }

    .recognition-label {
        color: #9da5b3;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .recognition-text {
        font-size: 32px;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777f8c;
        font-size: 13px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #252a33;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "model/normalized_landmark_sign_language_model.keras"
    )

    class_names = np.load(
        "model/normalized_class_names.npy",
        allow_pickle=True
    )

    return model, class_names


model, class_names = load_model()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_hand(hand_landmarks):

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Wrist as reference point
    wrist = landmarks[0].copy()

    landmarks[:, 0] -= wrist[0]
    landmarks[:, 1] -= wrist[1]
    landmarks[:, 2] -= wrist[2]

    # Scale normalization
    xy_distances = np.sqrt(
        landmarks[:, 0] ** 2 +
        landmarks[:, 1] ** 2
    )

    max_distance = np.max(xy_distances)

    if max_distance > 0:
        landmarks[:, :2] /= max_distance

    return landmarks.flatten()


# ============================================================
# SHARED RECOGNITION STATE
# ============================================================

class SignLanguageProcessor(VideoProcessorBase):

    def __init__(self):

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.current_gesture = "No hand detected"
        self.confidence = 0.0

        self.recognized_text = ""

        self.last_prediction = None
        self.prediction_start_time = None
        self.last_added_gesture = None

        self.stable_time = 0.8
        self.min_confidence = 0.70


    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # Mirror camera
        image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb_image)

        current_time = time.time()

        # ----------------------------------------------------
        # NO HAND
        # ----------------------------------------------------

        if not results.multi_hand_landmarks:

            self.current_gesture = "No hand detected"
            self.confidence = 0.0

            self.last_prediction = None
            self.prediction_start_time = None

            # Allows repeated letters such as LL
            self.last_added_gesture = None

            cv2.putText(
                image,
                "Show your hand",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # ----------------------------------------------------
        # EXTRACT LANDMARKS
        # ----------------------------------------------------

        hands_data = []

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = normalize_hand(
                hand_landmarks
            )

            wrist_x = hand_landmarks.landmark[0].x

            hands_data.append(
                (wrist_x, landmarks)
            )

        # Sort hands from left to right
        hands_data.sort(
            key=lambda x: x[0]
        )

        # ----------------------------------------------------
        # CREATE 126 FEATURES
        # ----------------------------------------------------

        features = []

        for _, landmarks in hands_data[:2]:
            features.extend(landmarks)

        # Pad if only one hand
        while len(features) < 126:
            features.extend([0.0] * 63)

        features = np.array(
            features[:126],
            dtype=np.float32
        )

        features = features.reshape(
            1,
            -1
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            features,
            verbose=0
        )[0]

        predicted_index = np.argmax(prediction)

        confidence = float(
            prediction[predicted_index]
        )

        gesture = str(
            class_names[predicted_index]
        )

        self.confidence = confidence

        # ----------------------------------------------------
        # LOW CONFIDENCE
        # ----------------------------------------------------

        if confidence < self.min_confidence:

            self.current_gesture = "Uncertain"

            cv2.putText(
                image,
                "Uncertain",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

        # ----------------------------------------------------
        # STABLE GESTURE DETECTION
        # ----------------------------------------------------

        if gesture != self.last_prediction:

            self.last_prediction = gesture
            self.prediction_start_time = current_time

        else:

            elapsed = (
                current_time -
                self.prediction_start_time
            )

            if (
                elapsed >= self.stable_time
                and
                gesture != self.last_added_gesture
            ):

                self.recognized_text += gesture

                self.last_added_gesture = gesture

        self.current_gesture = gesture

        # ----------------------------------------------------
        # DISPLAY ON VIDEO
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (15, 15),
            (390, 115),
            (14, 17, 23),
            -1
        )

        cv2.putText(
            image,
            f"Gesture: {gesture}",
            (30, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Confidence: {confidence * 100:.1f}%",
            (30, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (210, 215, 225),
            2
        )

        # Recognized text
        cv2.rectangle(
            image,
            (15, image.shape[0] - 75),
            (image.shape[1] - 15, image.shape[0] - 15),
            (14, 17, 23),
            -1
        )

        display_text = self.recognized_text

        if len(display_text) > 45:
            display_text = display_text[-45:]

        cv2.putText(
            image,
            f"Text: {display_text}",
            (30, image.shape[0] - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤟 Real-Time Sign Language Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Convert hand gestures into text using your webcam'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="info-card">
        <h3>📷 1. Camera</h3>
        <p>
        Your webcam captures the hand gesture in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="info-card">
        <h3>✋ 2. Hand Detection</h3>
        <p>
        Hand landmarks are detected from the camera frame.
        </p>
    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="info-card">
        <h3>🧠 3. Recognition</h3>
        <p>
        The trained model identifies the performed gesture.
        </p>
    </div>
    """, unsafe_allow_html=True)


with col4:

    st.markdown("""
    <div class="info-card">
        <h3>📝 4. Text</h3>
        <p>
        Stable gestures are converted into readable text.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CAMERA SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🎥 Live Recognition</div>',
    unsafe_allow_html=True
)


# Camera permission information

st.markdown("""
<div class="permission-box">

<div class="permission-title">
🔒 Camera Permission
</div>

<div class="permission-text">

Click <b>START</b> below to activate the camera.
Your browser will ask for camera permission if access has
not already been granted.

<br><br>

If you previously allowed camera access, your browser may
not show the permission popup again. You can change the
camera permission from your browser's site settings.

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# WEBRTC CAMERA
# ============================================================

ctx = webrtc_streamer(

    key="sign-language-recognition",

    video_processor_factory=SignLanguageProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True,

    media_toggle_controls=False
)


# ============================================================
# INSTRUCTIONS
# ============================================================

st.markdown(
    '<div class="section-title">📌 How to Use</div>',
    unsafe_allow_html=True
)

instruction_col1, instruction_col2 = st.columns(2)

with instruction_col1:

    st.markdown("""
    <div class="info-card">

    <h3>Using the camera</h3>

    <p>
    1. Click <b>START</b>.<br>
    2. Allow camera access when your browser asks.<br>
    3. Keep your hand clearly visible.<br>
    4. Hold a gesture steady for a moment.<br>
    5. The recognized letter will appear on screen.
    </p>

    </div>
    """, unsafe_allow_html=True)


with instruction_col2:

    st.markdown("""
    <div class="info-card">

    <h3>Building text</h3>

    <p>
    Hold each gesture until it is recognized.<br><br>
    Remove your hand briefly before repeating the
    same letter.<br><br>
    The recognized letters are displayed at the
    bottom of the camera.
    </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CONTROLS
# ============================================================

st.markdown(
    '<div class="section-title">⌨️ Controls</div>',
    unsafe_allow_html=True
)

control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:

    st.markdown("""
    **START**

    Start the webcam and begin recognition.
    """)

with control_col2:

    st.markdown("""
    **STOP**

    Stop the webcam when you are finished.
    """)

with control_col3:

    st.markdown("""
    **Remove hand**

    Remove your hand briefly before repeating
    the same gesture.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

Real-Time Sign Language Recognition<br>
Hand Gesture → Recognition → Text

</div>
""", unsafe_allow_html=True)

# ============================================
# SIGN REFERENCE
# ============================================

st.markdown("---")

st.header("Sign Reference")

st.write("Use these reference signs when practicing A–Z gestures.")

reference_path = os.path.join(
    os.path.dirname(__file__),
    "sign_reference"
)

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

cols = st.columns(5)

for i, letter in enumerate(letters):

    image_path = os.path.join(
        reference_path,
        f"{letter}.jpg"
    )

    with cols[i % 5]:

        if os.path.exists(image_path):

            st.image(
                image_path,
                caption=letter,
                use_container_width=True
            )

        else:
            st.write(f"{letter}")