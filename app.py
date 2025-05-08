import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.title("🎨 Minimal Canvas Background Test")

img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_file:
    try:
        original_image = Image.open(img_file).convert("RGB")
        img_array = np.array(original_image)
        height, width = img_array.shape[:2]

        st.image(original_image, caption="Uploaded Image Preview")

        st.subheader("Canvas with Background Image")

        # ✅ Force a fresh PIL Image from NumPy array (Cloud-friendly)
        background_image = Image.fromarray(img_array)

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#e00",
            background_image=background_image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="line",
            key="canvas_bg_test",
        )

    except Exception as e:
        st.error(f"Image loading or canvas error: {e}")
