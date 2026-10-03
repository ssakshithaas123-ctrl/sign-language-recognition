import os
import shutil

# ============================================
# DATASET PATH
# ============================================

DATASET_PATH = r"C:\Users\SAKSHI\Desktop\archive (5)\data"

# Where reference images will be saved
OUTPUT_PATH = r"C:\Users\SAKSHI\Desktop\sign-language-project\frontend\sign_reference"


# ============================================
# ALPHABET CLASSES
# ============================================

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ============================================
# CREATE OUTPUT FOLDER
# ============================================

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================
# EXTRACT ONE IMAGE FROM EACH LETTER
# ============================================

for letter in letters:

    class_folder = os.path.join(DATASET_PATH, letter)

    if not os.path.isdir(class_folder):
        print(f"❌ Folder not found: {letter}")
        continue

    images = [
        file for file in os.listdir(class_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not images:
        print(f"❌ No images found for: {letter}")
        continue

    # Sort so we consistently select the same image
    images.sort()

    selected_image = images[0]

    source = os.path.join(class_folder, selected_image)
    destination = os.path.join(OUTPUT_PATH, f"{letter}.jpg")

    shutil.copy2(source, destination)

    print(f"✅ {letter} -> {selected_image}")


print()
print("============================================")
print("SIGN REFERENCE CREATION COMPLETE")
print("============================================")
print(f"26 reference images saved to:")
print(OUTPUT_PATH)