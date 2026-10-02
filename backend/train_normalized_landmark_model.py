import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# LOAD NORMALIZED DATASET
# ==========================================

data = np.load("landmark_dataset_normalized.npz")

X = data["X"]
y = data["y"]
class_names = data["class_names"]

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", len(class_names))


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print()
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# ==========================================
# MLP MODEL
# ==========================================

model = models.Sequential([
    layers.Input(shape=(126,)),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(64, activation="relu"),

    layers.Dense(35, activation="softmax")
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_normalized_landmark_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)


# ==========================================
# TRAIN
# ==========================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)


# ==========================================
# TEST
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print()
print("==============================")
print("FINAL TEST RESULT")
print("==============================")

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    "normalized_landmark_sign_language_model.keras"
)


# ==========================================
# SAVE CLASS NAMES
# ==========================================

np.save(
    "normalized_class_names.npy",
    class_names
)


print()
print("==============================")
print("FILES SAVED")
print("==============================")

print(
    "normalized_landmark_sign_language_model.keras"
)

print(
    "normalized_class_names.npy"
)