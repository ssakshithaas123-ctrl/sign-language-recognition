import os
import cv2
import mediapipe as mp
import numpy as np


# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = r"C:\Users\SAKSHI\Desktop\archive (5)\data"

OUTPUT_FILE = "landmark_dataset_normalized.npz"


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)


# ==========================================
# CLASS NAMES
# ==========================================

class_names = sorted(
    [
        folder
        for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
    ],
    key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)
)

print("Classes:")
print(class_names)
print("Number of classes:", len(class_names))


# ==========================================
# FUNCTION:
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

    # Move wrist to (0, 0, 0)
    points = points - wrist

    # Calculate hand size using XY distance
    distances = np.sqrt(
        points[:, 0] ** 2 +
        points[:, 1] ** 2
    )

    scale = np.max(distances)

    if scale < 1e-6:
        scale = 1.0

    # Scale the entire hand
    points = points / scale

    return points.flatten()


# ==========================================
# EXTRACT DATA
# ==========================================

X = []
y = []

total_images = 0
successful_images = 0
failed_images = 0


for class_index, class_name in enumerate(class_names):

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    print()
    print(
        f"Processing class {class_index + 1}/{len(class_names)}: {class_name}"
    )

    image_files = [
        file
        for file in os.listdir(class_folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for image_file in image_files:

        total_images += 1

        image_path = os.path.join(
            class_folder,
            image_file
        )

        image = cv2.imread(image_path)

        if image is None:
            failed_images += 1
            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            failed_images += 1
            continue

        detected_hands = list(
            results.multi_hand_landmarks
        )

        # ==========================================
        # SORT HANDS BY WRIST X POSITION
        # ==========================================

        detected_hands.sort(
            key=lambda hand:
            hand.landmark[0].x
        )

        features = []

        # ==========================================
        # NORMALIZE EACH DETECTED HAND
        # ==========================================

        for hand in detected_hands[:2]:

            normalized = normalize_hand(hand)

            features.extend(normalized)

        # ==========================================
        # IF ONLY ONE HAND:
        # PAD SECOND HAND WITH ZEROS
        # ==========================================

        while len(features) < 126:
            features.extend([0.0] * 63)

        features = features[:126]

        X.append(features)
        y.append(class_index)

        successful_images += 1


# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)


# ==========================================
# RESULTS
# ==========================================

print()
print("==============================")
print("EXTRACTION COMPLETE")
print("==============================")

print("Total images checked:", total_images)
print("Successful images:", successful_images)
print("Failed images:", failed_images)

print("X shape:", X.shape)
print("y shape:", y.shape)

print()
print("Expected features per image: 126")


# ==========================================
# SAVE
# ==========================================

np.savez(
    OUTPUT_FILE,
    X=X,
    y=y,
    class_names=np.array(class_names)
)

print()
print("Saved file:", OUTPUT_FILE)

hands.close()