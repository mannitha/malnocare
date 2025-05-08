import streamlit as st
from PIL import Image
import numpy as np
import math
import pandas as pd

st.title("📏 Height Estimation Using Steel Scale Reference")

# Initialize session state for clicks
if "clicks" not in st.session_state:
    st.session_state.clicks = []

img_file = st.file_uploader("Upload full-body image (with visible steel scale)", type=["jpg", "jpeg", "png"])

if img_file:
    image = Image.open(img_file).convert("RGB")
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    st.image(image, caption="Click on two ends of the scale, then top of head and bottom of feet", use_column_width=False)

    # Get clicks from user
    click = st.image(image, caption="Click points in order: [top scale, bottom scale, head, feet]", use_column_width=False)

    if st.button("Reset Points"):
        st.session_state.clicks = []

    coords = st.experimental_data_editor(
        pd.DataFrame(columns=["X", "Y"], data=st.session_state.clicks),
        num_rows="dynamic",
        use_container_width=True,
        key="coords_editor"
    )

    # Add instructions
    st.info("Click points in this order: 🔹Top of Scale → 🔹Bottom of Scale → 🔹Top of Head → 🔹Bottom of Feet")

    # Use uploaded click coordinates manually if you're building this as custom HTML/JS
    # For now, simulate by asking for manual input:
    if len(st.session_state.clicks) < 4:
        st.warning("You need to provide 4 coordinate points in total.")
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("X", 0, width, step=1, key=f"x{len(st.session_state.clicks)}")
        with col2:
            y = st.number_input("Y", 0, height, step=1, key=f"y{len(st.session_state.clicks)}")

        if st.button("Add Point"):
            st.session_state.clicks.append((x, y))

    if len(st.session_state.clicks) == 4:
        (sx1, sy1), (sx2, sy2), (hx, hy), (fx, fy) = st.session_state.clicks

        def euclidean(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        scale_pixels = euclidean((sx1, sy1), (sx2, sy2))
        body_pixels = euclidean((hx, hy), (fx, fy))

        reference_cm = 32.0
        cm_per_pixel = reference_cm / scale_pixels
        estimated_height = body_pixels * cm_per_pixel

        st.success(f"Scale pixel distance: {scale_pixels:.2f} px")
        st.success(f"Body pixel distance: {body_pixels:.2f} px")
        st.success(f"Estimated Height: **{estimated_height:.2f} cm**")

