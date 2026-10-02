import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    "model/normalized_landmark_sign_language_model.keras"
)

class_names = np.load(
    "model/normalized_class_names.npy",
    allow_pickle=True
)


# ============================================================
# 2. MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# 3. NORMALIZE HAND LANDMARKS
# ============================================================

def normalize_hand(hand_landmarks):

    points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks.landmark
        ],
        dtype=np.float32
    )

    # Wrist = landmark 0
    wrist = points[0].copy()

    # Move wrist to origin
    points = points - wrist

    # Calculate hand size
    distances = np.sqrt(
        points[:, 0] ** 2 +
        points[:, 1] ** 2
    )

    scale = np.max(distances)

    if scale < 1e-6:
        scale = 1.0

    # Scale hand
    points = points / scale

    return points.flatten()


# ============================================================
# 4. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# 5. TEXT VARIABLES
# ============================================================

text = ""

# Current gesture being observed
current_gesture = ""

# When the current gesture first appeared
gesture_start_time = 0

# Last letter that was added
last_added_gesture = ""

# How long gesture must stay stable
STABLE_TIME = 0.8

# Minimum confidence
MIN_CONFIDENCE = 70


# ============================================================
# 6. START MESSAGE
# ============================================================

print("==========================================")
print(" SIGN LANGUAGE TO TEXT")
print("==========================================")
print()
print("Show a letter to the camera.")
print("Keep the gesture steady for about 1 second.")
print()
print("Controls:")
print("Q     = Quit")
print("C     = Clear text")
print("SPACE = Add space")
print()
print("==========================================")


# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect hands
    results = hands.process(rgb)


    # Default values
    prediction_text = "No hand"
    confidence = 0.0


    # ========================================================
    # 8. IF HAND IS DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        detected_hands = list(
            results.multi_hand_landmarks
        )

        # Sort hands from left to right
        detected_hands.sort(
            key=lambda hand:
            hand.landmark[0].x
        )


        # ----------------------------------------------------
        # Extract normalized landmarks
        # ----------------------------------------------------

        features = []

        for hand in detected_hands[:2]:

            normalized = normalize_hand(hand)

            features.extend(normalized)


        # If only one hand exists,
        # add zeros for the second hand
        while len(features) < 126:

            features.extend(
                [0.0] * 63
            )


        # Make sure exactly 126 features
        features = features[:126]


        # Convert to NumPy
        X = np.array(
            features,
            dtype=np.float32
        )

        X = np.expand_dims(
            X,
            axis=0
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            X,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions[0]
        )

        prediction_text = str(
            class_names[predicted_index]
        )

        confidence = (
            predictions[0][predicted_index]
            * 100
        )


        # ====================================================
        # 9. STABLE GESTURE CHECK
        # ====================================================

        current_time = time.time()


        # Only use alphabet letters
        if (
            prediction_text.isalpha()
            and confidence >= MIN_CONFIDENCE
        ):

            # New gesture detected
            if prediction_text != current_gesture:

                current_gesture = prediction_text

                gesture_start_time = current_time


            # Same gesture continues
            else:

                stable_duration = (
                    current_time
                    - gesture_start_time
                )


                # Gesture has been stable long enough
                if (
                    stable_duration >= STABLE_TIME
                    and prediction_text != last_added_gesture
                ):

                    text += prediction_text

                    last_added_gesture = prediction_text

                    print(
                        "Added:",
                        prediction_text
                    )

                    # Reset timer
                    gesture_start_time = current_time


        else:

            # Prediction is not reliable
            current_gesture = ""

            gesture_start_time = 0


        # ----------------------------------------------------
        # Draw hand landmarks
        # ----------------------------------------------------

        for hand in detected_hands:

            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


    # ========================================================
    # 10. NO HAND DETECTED
    # ========================================================

    else:

        prediction_text = "No hand"

        confidence = 0.0

        # Reset current gesture
        current_gesture = ""

        gesture_start_time = 0

        # IMPORTANT:
        # This allows the same letter to be entered again.
        #
        # Example:
        # L → remove hand → L
        #
        # This allows "LL".


        last_added_gesture = ""


    # ========================================================
    # 11. DISPLAY GESTURE
    # ========================================================

    cv2.putText(
        frame,
        f"Gesture: {prediction_text}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )


    # ========================================================
    # 12. DISPLAY CONFIDENCE
    # ========================================================

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ========================================================
    # 13. DISPLAY STABILITY TIMER
    # ========================================================

    if (
        current_gesture
        and confidence >= MIN_CONFIDENCE
    ):

        stable_duration = (
            time.time()
            - gesture_start_time
        )

        progress = min(
            stable_duration / STABLE_TIME,
            1.0
        )

        progress_text = (
            f"Hold: {progress * 100:.0f}%"
        )

        cv2.putText(
            frame,
            progress_text,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # ========================================================
    # 14. TEXT BOX
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 135),
        (frame.shape[1] - 10, 200),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Text: {text}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 15. CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "C = Clear | SPACE = Space | Q = Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 16. SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "Sign Language To Text",
        frame
    )


    # ========================================================
    # 17. KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == ord("q"):

        break


    # Clear text
    elif key == ord("c"):

        text = ""

        current_gesture = ""

        last_added_gesture = ""

        gesture_start_time = 0

        print("Text cleared.")


    # Space
    elif key == 32:

        text += " "

        print("Added: SPACE")


# ============================================================
# 18. CLEANUP
# ============================================================

cap.release()

hands.close()

cv2.destroyAllWindows()


# ============================================================
# 19. FINAL TEXT
# ============================================================

print()
print("==========================================")
print("FINAL RECOGNIZED TEXT:")
print(text)
print("==========================================")