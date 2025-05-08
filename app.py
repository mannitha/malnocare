import streamlit as st
from PIL import Image
import numpy as np
import math
from streamlit_image_coordinates import st_image_coordinates

# Function to calculate Euclidean distance between two points
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Initialize session state to track coordinates
if "clicks" not in st.session_state:
    st.session_state.clicks = []

st.title("📏 Height Estimation Using Steel Scale Reference")

# Upload image
img_file = st.file_uploader("Upload full-body image (with visible steel scale)", type=["jpg", "jpeg", "png"])

if img_file:
    # Open and display the image
    image = Image.open(img_file).convert("RGB")
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    st.image(image, caption="Click on the image to select points", use_column_width=True)

    # Capture coordinates when the user clicks on the image
    coords = st_image_coordinates(image)

    if coords:
        # Add new points to the session state clicks list
        st.session_state.clicks.append(coords)

        # Display the clicked points
        st.write("Clicked points:", st.session_state.clicks)

        # Check if 4 points are selected: top of scale, bottom of scale, head, and feet
        if len(st.session_state.clicks) == 4:
            # Extract points
            (sx1, sy1), (sx2, sy2), (hx, hy), (fx, fy) = st.session_state.clicks

            # Calculate pixel distances
            scale_pixels = euclidean((sx1, sy1), (sx2, sy2))
            body_pixels = euclidean((hx, hy), (fx, fy))

            # Known scale length (in cm)
            reference_cm = 32.0
            cm_per_pixel = reference_cm / scale_pixels
            estimated_height = body_pixels * cm_per_pixel

            # Display results
            st.success(f"Scale pixel distance: {scale_pixels:.2f} px")
            st.success(f"Body pixel distance: {body_pixels:.2f} px")
            st.success(f"Estimated Height: **{estimated_height:.2f} cm**")

    else:
        st.warning("Click on the image to select points.")
