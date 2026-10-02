import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(
    "landmark_sign_language_model.keras"
)

data = np.load("landmark_normalization.npz")

mean = data["mean"]
std = data["std"]
class_names = data["class_names"]


# ==============================
# MEDIAPIPE HANDS
# ==============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==============================
# WEBCAM
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print("Webcam started.")
print("Show a sign to the camera.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb)


    # ==============================
    # EXTRACT 126 LANDMARK FEATURES
    # ==============================

    features = []

    if results.multi_hand_landmarks:

        # Maximum 2 hands
        detected_hands = results.multi_hand_landmarks[:2]

        for hand_landmarks in detected_hands:

            for landmark in hand_landmarks.landmark:
                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

        # If only one hand detected,
        # add zeros for the second hand
        while len(features) < 126:
            features.extend([0.0, 0.0, 0.0])

        # Keep exactly 126 values
        features = features[:126]

        # Convert to numpy
        X = np.array(features, dtype=np.float32)

        # Apply SAME normalization used during training
        X = (X - mean) / std

        # Add batch dimension
        X = np.expand_dims(X, axis=0)


        # ==============================
        # PREDICTION
        # ==============================

        predictions = model.predict(X, verbose=0)

        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100


        # ==============================
        # DISPLAY PREDICTION
        # ==============================

        text = f"Prediction: {predicted_class}"
        confidence_text = f"Confidence: {confidence:.2f}%"

        cv2.putText(
            frame,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            confidence_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ==============================
        # DRAW HAND LANDMARKS
        # ==============================

        for hand_landmarks in detected_hands:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
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


    # ==============================
    # SHOW FRAME
    # ==============================

    cv2.imshow(
        "Sign Language Recognition - Landmark Model",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Webcam stopped.")