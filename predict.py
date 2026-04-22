import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load best model
model = tf.keras.models.load_model("best_model.h5")

# Class names (VERY IMPORTANT)
class_names = ['cow', 'buffalo']

# Load image
img_path = "test.jpg"   # change this to your image name
img = image.load_img(img_path, target_size=(224, 224))

# Preprocess
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

# Output
if prediction[0][0] > 0.5:
    print("COW")
else:
    print("BUFFALO")