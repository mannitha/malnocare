#streamlit whitel blank
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from streamlit_drawable_canvas import st_canvas

mp_pose = mp.solutions.pose

def load_image(uploaded_file):
    img = Image.open(uploaded_file)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_keypoints(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            h, w, _ = image.shape
            landmarks = results.pose_landmarks.landmark
            head_y = int(landmarks[mp_pose.PoseLandmark.NOSE].y * h)
            foot_left_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)
            foot_right_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h)
            foot_y = max(foot_left_y, foot_right_y)
            return head_y, foot_y
    return None, None

def draw_landmarks(image, head_y, foot_y):
    annotated = image.copy()
    center_x = image.shape[1] // 2
    cv2.line(annotated, (center_x, head_y), (center_x, foot_y), (0,255,0), 2)
    cv2.circle(annotated, (center_x, head_y), 5, (255,0,0), -1)
    cv2.circle(annotated, (center_x, foot_y), 5, (0,0,255), -1)
    return annotated

def get_pixel_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def run_height_estimator():
    st.title("📏 Height Estimator from Single Image")
    st.markdown("Upload a full-body image **with a visible reference object**, and specify its real-world length.")

    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    
    if img_file:
        image = Image.open(img_file)
        img_np = np.array(image)

        reference_length = st.number_input("Enter the real-world length of the reference object (in cm)", min_value=1.0, step=0.5)

        st.subheader("Step 1: Draw a line over the reference object")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#e00",
            background_image=image,
            update_streamlit=True,
            height=img_np.shape[0],
            width=img_np.shape[1],
            drawing_mode="line",
            key="canvas",
        )

        if canvas_result.json_data and "objects" in canvas_result.json_data:
            objs = canvas_result.json_data["objects"]
            if len(objs) >= 1 and objs[0]["type"] == "line":
                line = objs[0]
                x1, y1 = line["x1"], line["y1"]
                x2, y2 = line["x2"], line["y2"]
                pixel_dist = get_pixel_distance((x1, y1), (x2, y2))
                calibration_factor = reference_length / pixel_dist  # user-defined cm / pixel

                st.success(f"Calibration complete: {calibration_factor:.4f} cm/pixel")

                st.subheader("Step 2: Estimating height from landmarks")
                image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                head_y, foot_y = detect_keypoints(image_bgr)

                if head_y is not None and foot_y is not None:
                    pixel_height = abs(foot_y - head_y)
                    estimated_height = calibration_factor * pixel_height
                    annotated_img = draw_landmarks(image_bgr, head_y, foot_y)
                    st.image(annotated_img, caption="Detected Height", channels="BGR")
                    st.success(f"Estimated Height: **{estimated_height:.2f} cm**")
                else:
                    st.error("❌ Could not detect body landmarks. Please try a clearer full-body image.")
            else:
                st.info("Draw a line over the known-length reference object.")
        else:
            st.info("Draw a line to calibrate using the reference object.")

if __name__ == "__main__":
    run_height_estimator()
