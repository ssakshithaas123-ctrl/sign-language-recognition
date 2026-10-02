import os
import cv2
import mediapipe as mp

DATASET_DIR = r"C:\Users\SAKSHI\Desktop\archive (5)\data"

# Pick one image from class A
class_folder = os.path.join(DATASET_DIR, "A")

image_files = os.listdir(class_folder)
image_path = os.path.join(class_folder, image_files[0])

print("Testing image:")
print(image_path)

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image.")
    exit()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        print()
        print("==============================")
        print("SUCCESS")
        print("Hands detected:", len(results.multi_hand_landmarks))

        for i, hand in enumerate(results.multi_hand_landmarks):
            print(
                f"Hand {i + 1}: "
                f"{len(hand.landmark)} landmarks"
            )

        print("==============================")

    else:
        print()
        print("==============================")
        print("NO HAND DETECTED")
        print("==============================")