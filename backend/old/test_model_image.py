import tensorflow as tf
import numpy as np
import cv2


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    "model/sign_language_model.keras"
)

print("Model loaded successfully.")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
]


# ============================================================
# LOAD DATASET IMAGE
# ============================================================

image_path = "webcam_A_test.jpg"
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not find the image.")
    print("Make sure test_A.jpg is in the project folder.")
    exit()


# ============================================================
# RESIZE TO 128 x 128
# ============================================================

image = cv2.resize(
    image,
    (128, 128)
)


# ============================================================
# BGR -> RGB
# ============================================================

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# NORMALIZE
# ============================================================

image = image.astype(
    np.float32
) / 255.0


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image = np.expand_dims(
    image,
    axis=0
)


# ============================================================
# PREDICT
# ============================================================

predictions = model.predict(
    image,
    verbose=0
)

predicted_index = np.argmax(
    predictions[0]
)

predicted_class = class_names[
    predicted_index
]

confidence = predictions[0][
    predicted_index
]


# ============================================================
# RESULT
# ============================================================

print()
print("==============================")
print("MODEL TEST RESULT")
print("==============================")
print("Expected: A")
print("Predicted:", predicted_class)
print(
    f"Confidence: {confidence * 100:.2f}%"
)
print("==============================")