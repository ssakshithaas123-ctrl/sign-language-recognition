import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# ============================================================
# 1. LOAD TRAINED CNN MODEL
# ============================================================

model = tf.keras.models.load_model(
    "model/sign_language_model.keras"
)

print("CNN model loaded successfully.")


# ============================================================
# 2. CLASS NAMES
# ============================================================

class_names = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
]


# ============================================================
# 3. INITIALIZE MEDIAPIPE
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
# 4. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Real-time sign recognition started.")
print("Press Q to quit.")


# ============================================================
# 5. MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # --------------------------------------------------------
    # Keep clean copy for CNN
    # --------------------------------------------------------

    clean_frame = frame.copy()

    height, width, _ = frame.shape


    # ========================================================
    # 6. CONVERT BGR -> RGB FOR MEDIAPIPE
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)


    # ========================================================
    # 7. IF HAND(S) ARE DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        all_x = []
        all_y = []


        # ----------------------------------------------------
        # Get landmarks from all detected hands
        # ----------------------------------------------------

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks on webcam display
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            # ------------------------------------------------
            # Convert normalized coordinates to pixels
            # ------------------------------------------------

            for landmark in hand_landmarks.landmark:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                all_x.append(x)
                all_y.append(y)


        # ====================================================
        # 8. FIND OVERALL HAND BOUNDING BOX
        # ====================================================

        x_min = max(
            0,
            min(all_x) - 20
        )

        y_min = max(
            0,
            min(all_y) - 20
        )

        x_max = min(
            width,
            max(all_x) + 20
        )

        y_max = min(
            height,
            max(all_y) + 20
        )


        # ====================================================
        # 9. CROP HAND REGION
        # ====================================================

        crop = clean_frame[
            y_min:y_max,
            x_min:x_max
        ]


        # Make sure crop is valid

        if crop.size > 0:

            # =================================================
            # 10. CREATE BLACK MASK
            # =================================================

            mask = np.zeros(
                crop.shape[:2],
                dtype=np.uint8
            )


            # =================================================
            # 11. CREATE HAND SHAPE USING LANDMARKS
            # =================================================

            for hand_landmarks in results.multi_hand_landmarks:

                points = []

                for landmark in hand_landmarks.landmark:

                    # Convert landmark to original image pixels

                    px = int(
                        landmark.x * width
                    )

                    py = int(
                        landmark.y * height
                    )


                    # Convert to crop coordinates

                    px = px - x_min
                    py = py - y_min


                    # Keep coordinates inside crop

                    px = max(
                        0,
                        min(
                            crop.shape[1] - 1,
                            px
                        )
                    )

                    py = max(
                        0,
                        min(
                            crop.shape[0] - 1,
                            py
                        )
                    )

                    points.append(
                        [px, py]
                    )


                # Convert points to NumPy array

                points = np.array(
                    points,
                    dtype=np.int32
                )


                # Create convex hull

                hull = cv2.convexHull(
                    points
                )


                # Fill hand area

                cv2.fillConvexPoly(
                    mask,
                    hull,
                    255
                )


            # =================================================
            # 12. KEEP ONLY MASKED HAND REGION
            # =================================================

            clean_gesture = cv2.bitwise_and(
                crop,
                crop,
                mask=mask
            )


            # =================================================
            # 13. RESIZE TO 128 x 128
            # =================================================

            gesture_128 = cv2.resize(
                clean_gesture,
                (128, 128)
            )


            # =================================================
            # 14. CONVERT BGR -> RGB
            # =================================================

            gesture_rgb = cv2.cvtColor(
                gesture_128,
                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # 15. NORMALIZE IMAGE
            # =================================================

            gesture_normalized = (
                gesture_rgb.astype(
                    np.float32
                ) / 255.0
            )


            # =================================================
            # 16. ADD BATCH DIMENSION
            # =================================================

            input_image = np.expand_dims(
                gesture_normalized,
                axis=0
            )


            # =================================================
            # 17. CNN PREDICTION
            # =================================================

            predictions = model.predict(
                input_image,
                verbose=0
            )


            # Find highest probability class

            predicted_index = np.argmax(
                predictions[0]
            )


            predicted_class = class_names[
                predicted_index
            ]


            confidence = predictions[0][
                predicted_index
            ]


            # =================================================
            # 18. DISPLAY PREDICTION
            # =================================================

            cv2.putText(
                frame,
                f"Prediction: {predicted_class}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )


            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )


            # =================================================
            # 19. DRAW HAND BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )


            # =================================================
            # 20. SHOW ACTUAL CNN INPUT
            # =================================================

            display_crop = cv2.resize(
                gesture_128,
                (256, 256)
            )


            cv2.imshow(
                "CNN Input - 128x128",
                display_crop
            )


            # =================================================
            # 21. SAVE CNN INPUT FOR TESTING
            # =================================================

            cv2.imwrite(
                "webcam_A_test.jpg",
                gesture_128
            )


            # Move CNN input window

            cv2.moveWindow(
                "CNN Input - 128x128",
                950,
                50
            )


    # ========================================================
    # 22. NO HAND DETECTED
    # ========================================================

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # 23. SHOW MAIN WEBCAM
    # ========================================================

    cv2.imshow(
        "Real-Time Sign Language Recognition",
        frame
    )


    # ========================================================
    # 24. QUIT WITH Q
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 25. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Program stopped.")