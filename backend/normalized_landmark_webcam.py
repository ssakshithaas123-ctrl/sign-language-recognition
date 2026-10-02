import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "normalized_landmark_sign_language_model.keras"
)

class_names = np.load(
    "normalized_class_names.npy",
    allow_pickle=True
)


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# NORMALIZE ONE HAND
# ==========================================

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

    # Scale normalize
    points = points / scale

    return points.flatten()


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print("Webcam started.")
print("Show a sign to the camera.")
print("Press Q to quit.")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect hands
    results = hands.process(rgb)


    # ==========================================
    # EXTRACT FEATURES
    # ==========================================

    if results.multi_hand_landmarks:

        detected_hands = list(
            results.multi_hand_landmarks
        )

        # Same ordering as training:
        # leftmost hand → first
        # rightmost hand → second

        detected_hands.sort(
            key=lambda hand:
            hand.landmark[0].x
        )

        features = []

        # Normalize each hand
        for hand in detected_hands[:2]:

            normalized = normalize_hand(hand)

            features.extend(normalized)

        # If only one hand detected,
        # pad second hand with zeros
        while len(features) < 126:
            features.extend([0.0] * 63)

        features = features[:126]

        X = np.array(
            features,
            dtype=np.float32
        )

        X = np.expand_dims(
            X,
            axis=0
        )


        # ==========================================
        # PREDICTION
        # ==========================================

        predictions = model.predict(
            X,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            predictions[0][predicted_index] * 100
        )


        # ==========================================
        # DISPLAY
        # ==========================================

        cv2.putText(
            frame,
            f"Prediction: {predicted_class}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ==========================================
        # DRAW LANDMARKS
        # ==========================================

        for hand in detected_hands:

            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )


    # ==========================================
    # SHOW
    # ==========================================

    cv2.imshow(
        "Normalized Sign Language Recognition",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Webcam stopped.")