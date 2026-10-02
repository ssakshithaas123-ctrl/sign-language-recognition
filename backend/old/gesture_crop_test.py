import cv2
import mediapipe as mp

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Square gesture crop test started.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    height, width, _ = frame.shape

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        all_x = []
        all_y = []

        # Process every detected hand
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Collect coordinates
            for landmark in hand_landmarks.landmark:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                all_x.append(x)
                all_y.append(y)

        # Find complete bounding box
        x_min = min(all_x)
        y_min = min(all_y)
        x_max = max(all_x)
        y_max = max(all_y)

        # Padding
        padding = 40

        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding

        # Keep coordinates inside image
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        # -----------------------------------
        # MAKE THE CROP SQUARE
        # -----------------------------------

        crop_width = x_max - x_min
        crop_height = y_max - y_min

        square_size = max(crop_width, crop_height)

        # Center of bounding box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # New square coordinates
        square_x_min = center_x - square_size // 2
        square_y_min = center_y - square_size // 2

        square_x_max = square_x_min + square_size
        square_y_max = square_y_min + square_size

        # Adjust if square goes outside frame
        if square_x_min < 0:
            square_x_max -= square_x_min
            square_x_min = 0

        if square_y_min < 0:
            square_y_max -= square_y_min
            square_y_min = 0

        if square_x_max > width:
            difference = square_x_max - width
            square_x_min -= difference
            square_x_max = width

        if square_y_max > height:
            difference = square_y_max - height
            square_y_min -= difference
            square_y_max = height

        # Final safety
        square_x_min = max(0, square_x_min)
        square_y_min = max(0, square_y_min)
        square_x_max = min(width, square_x_max)
        square_y_max = min(height, square_y_max)

        # Draw square bounding box
        cv2.rectangle(
            frame,
            (square_x_min, square_y_min),
            (square_x_max, square_y_max),
            (0, 255, 0),
            2
        )

        # -----------------------------------
        # CROP COMPLETE GESTURE
        # -----------------------------------

        gesture_crop = frame[
            square_y_min:square_y_max,
            square_x_min:square_x_max
        ]

        if gesture_crop.size > 0:

            # Resize to CNN input size
            gesture_128 = cv2.resize(
                gesture_crop,
                (128, 128)
            )

            # Show resized image
            cv2.imshow(
                "128x128 CNN Input",
                gesture_128
            )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Original webcam
    cv2.imshow(
        "Webcam - Gesture Detection",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()