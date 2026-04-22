import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Page config
st.set_page_config(page_title="Animal Classifier", page_icon="🐄", layout="wide")

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.title("🐄🐃 Smart Animal Classifier")
st.write("Upload or capture an image to classify Cow vs Buffalo")

st.write("---")

# Sidebar
st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

st.sidebar.write("📊 Model Accuracy: ~90%")
st.sidebar.write("🧠 Model: CNN")

# 🧠 Explanation panel
with st.sidebar.expander("🧠 How it works"):
    st.write("""
    The model analyzes:
    - Body shape  
    - Horn structure  
    - Skin color  
    - Texture patterns  
    """)

# 📸 Camera input
camera_img = st.camera_input("📸 Take a photo")

# 📤 File upload
uploaded_files = st.file_uploader(
    "📤 Upload images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# Combine inputs
images = []

if camera_img:
    images.append(("camera.jpg", camera_img))

if uploaded_files:
    for file in uploaded_files:
        images.append((file.name, file))

# Layout
col1, col2 = st.columns(2)

for name, file in images:
    img = Image.open(file).convert("RGB")
    img_resized = img.resize((224, 224))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > threshold:
        label = "cow"
        confidence = prediction * 100
    else:
        label = "buffalo"
        confidence = (1 - prediction) * 100

    # Store history
    st.session_state.history.append({
        "Image": name,
        "Prediction": label,
        "Confidence": round(confidence, 2)
    })

    with col1:
        st.image(img, caption=name, use_column_width=True)

    with col2:
        st.subheader(f"Prediction: {label}")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

        if confidence > 85:
            st.success("High Confidence ✅")
        elif confidence > 60:
            st.warning("Medium Confidence ⚠️")
        else:
            st.error("Low Confidence ❌")

        st.write("---")

# 📊 History section
st.write("## 📊 Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    # 📥 Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", csv, "results.csv", "text/csv")
else:
    st.write("No predictions yet.")

# 🧹 Clear history
if st.button("🧹 Clear History"):
    st.session_state.history = []