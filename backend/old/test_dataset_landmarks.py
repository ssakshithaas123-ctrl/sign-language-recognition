import cv2
import mediapipe as mp

# Load one dataset image
image = cv2.imread("test_A.jpg")

if image is None:
    print("ERROR: Could not load test_A.jpg")
    exit()

# Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# MediaPipe Hands
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        print("==============================")
        print("Hands detected:", len(results.multi_hand_landmarks))

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            print(f"Hand {i + 1}:")
            print("Number of landmarks:", len(hand_landmarks.landmark))

        print("==============================")
        print("SUCCESS: MediaPipe extracted hand landmarks.")
    else:
        print("==============================")
        print("NO HAND DETECTED")
        print("==============================")