import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Folder path
test_folder = "test_images"

# Counters
cow_count = 0
buffalo_count = 0

# Loop through images
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            result = "Cow 🐄"
            cow_count += 1
        else:
            result = "Buffalo 🐃"
            buffalo_count += 1

        print(f"{img_name} --> {result}")

    except Exception as e:
        print(f"Error in {img_name}: {e}")

# FINAL OUTPUT
print("\n📊 FINAL RESULT:")
print(f"🐄 Cow: {cow_count}")
print(f"🐃 Buffalo: {buffalo_count}")