import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

@st.cache_resource
def load_image(file):
    return Image.open(file).convert("RGB")

def main():
    st.title("📏 Canvas Image Test")

    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_file:
        image = load_image(img_file)

        # Resize if necessary
        max_width = 800
        if image.width > max_width:
            scale = max_width / image.width
            image = image.resize((max_width, int(image.height * scale)))

        # Display the image
        st.image(image, caption="Uploaded Image Preview", use_column_width=True)

        st.subheader("Step 1: Draw a line on the image")

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#e00",
            background_image=image,  # Using original loaded image
            update_streamlit=True,
            height=image.height,
            width=image.width,
            drawing_mode="line",
            key="canvas",
        )

if __name__ == "__main__":
    main()
