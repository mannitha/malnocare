import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("🧪 Canvas Image Debug")

img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if img_file:
    # Open and convert the image to RGB to ensure compatibility
    image = Image.open(img_file).convert("RGB")
    
    # Optional: Resize the image if it's too large
    max_width = 800
    if image.width > max_width:
        scale = max_width / image.width
        image = image.resize((max_width, int(image.height * scale)))
    
    # Convert the image to a NumPy array and back to a PIL Image to strip metadata
    img_np = np.array(image)
    image_for_canvas = Image.fromarray(img_np)

    # Display the image as a preview
    st.image(image_for_canvas, caption="Image Preview", use_column_width=True)

    st.subheader("Drawable Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#e00",
        background_image=image_for_canvas,
        update_streamlit=True,
        height=img_np.shape[0],
        width=img_np.shape[1],
        drawing_mode="line",
        key="canvas-test",
    )
