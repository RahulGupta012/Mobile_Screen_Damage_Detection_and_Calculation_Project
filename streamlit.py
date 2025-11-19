
import streamlit as st
import numpy as np
from PIL import Image
from Seg_img_damage_calculation import damage_pipeline


st.title("Screen Damage Detection")
st.write("Upload a photo of your phone screen to check for damage.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "avif"])

if uploaded_file:
    # Convert uploaded file to numpy array
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    st.image(image, caption="Your Uploaded Image", use_container_width=True)

    # Run the damage detection pipeline
    segmented, edges, damage_percent = damage_pipeline(img_array)

    if segmented is None:
        st.error("No phone detected in the image.")
    else:
        st.write(f"**Estimated Damage Area:** {damage_percent:.2f}%")
        st.image(segmented, caption="Segmented Phone Area", use_container_width=True)
        st.image(edges, caption="Detected Damage (Edges)", use_container_width=True)
