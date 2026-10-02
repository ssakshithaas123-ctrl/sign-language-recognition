import numpy as np

data = np.load("landmark_dataset.npz")

X = data["X"]
y = data["y"]
class_names = data["class_names"]

print("==============================")
print("LANDMARK DATASET")
print("==============================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Number of classes:", len(class_names))
print("Classes:")
print(class_names)

print()
print("First sample:")
print(X[0])

print()
print("First label:", y[0])
print("First class:", class_names[y[0]])

print("==============================")