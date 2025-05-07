import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from streamlit_drawable_canvas import st_canvas

# --- Session State Setup ---
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# --- Fake Login/Signup Flow ---
def login_screen():
    st.title("🔐 Login / Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.session_state.page = 'main_menu'
        else:
            st.error("Please enter both username and password.")

# --- Main Menu ---
def main_menu():
    st.title("🌿 Welcome to NutriAid")
    choice = st.radio("Choose a feature", ["Nutrition Detection", "NutriMann (Food Scanner)"])

    if choice == "Nutrition Detection":
        st.session_state.page = 'nutrition_menu'
    elif choice == "NutriMann (Food Scanner)":
        st.warning("🔧 NutriMann is coming soon!")

# --- Nutrition Menu ---
def nutrition_menu():
    st.title("📊 Nutrition Detection")
    choice = st.radio("Choose an option", ["New Data", "View Previous Data", "Modify Data", "Back"])

    if choice == "New Data":
        st.session_state.page = 'input_form'
    elif choice == "Back":
        st.session_state.page = 'main_menu'
    else:
        st.info("🔧 Feature not yet implemented.")

# --- Step 1: Input Form ---
def input_form():
    st.title("📝 Enter Child Info")
    name = st.text_input("Name")
    age = st.number_input("Age (years)", min_value=0.0)
    weight = st.number_input("Weight (kg)", min_value=0.0)

    if st.button("Next"):
        if name and age > 0 and weight > 0:
            st.session_state.user_data = {"name": name, "age": age, "weight": weight}
            st.session_state.page = 'height_step'
        else:
            st.error("Please fill all fields properly.")

# --- Step 2: Height Step ---
def height_step():
    st.title("📏 Estimate Height")

    SCALE_LENGTH_CM = 32
    mp_pose = mp.solutions.pose

    uploaded_file = st.file_uploader("Upload full-body image with 30cm steel scale", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h_img, w_img, _ = img.shape

        st.markdown("🟢 Click exactly two points on the scale.")
        from PIL import Image
import cv2
import numpy as np

# Assume img is your OpenCV image (NumPy array)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)  # This is correct

canvas_result = st_canvas(
    background_image=pil_img,  # must be a PIL.Image
    height=img.shape[0],
    width=img.shape[1],
    drawing_mode="point",
    point_display_radius=5,
    key="canvas",
)

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) == 2:
            points = [(int(obj["left"]), int(obj["top"])) for obj in canvas_result.json_data["objects"]]
            with mp_pose.Pose(static_image_mode=True) as pose:
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    head_y = int(landmarks[mp_pose.PoseLandmark.NOSE].y * h_img)
                    foot_y = max(
                        int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h_img),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h_img),
                    )
                    pixel_height = foot_y - head_y
                    scale_pixel_length = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
                    pixels_per_cm = scale_pixel_length / SCALE_LENGTH_CM
                    height_cm = pixel_height / pixels_per_cm

                    st.session_state.user_data["height"] = height_cm
                    st.success(f"✅ Estimated Height: {height_cm:.2f} cm")
                    if st.button("Next"):
                        st.session_state.page = 'arm_step'
                else:
                    st.error("❌ Pose landmarks not detected.")
        else:
            st.info("Please mark exactly 2 points on the steel scale.")

# --- Step 3: Arm Step ---
def arm_step():
    st.title("💪 Estimate MUAC")
    CALIBRATION_FACTOR = 0.09166
    mp_pose = mp.solutions.pose

    col1, col2 = st.columns(2)
    with col1:
        cam = st.button("📷 Camera")
    with col2:
        upload = st.button("🖼 Upload")

    mode = st.session_state.get("arm_input", None)

    if cam:
        st.session_state.arm_input = "camera"
    elif upload:
        st.session_state.arm_input = "upload"

    image = None
    if st.session_state.get("arm_input") == "camera":
        cam_data = st.camera_input("Take a picture")
        if cam_data:
            image = Image.open(cam_data)
    elif st.session_state.get("arm_input") == "upload":
        upload_file = st.file_uploader("Upload upper-body image", type=["jpg", "jpeg", "png"])
        if upload_file:
            image = Image.open(upload_file)

    if image:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                h, w, _ = img_cv.shape
                lm = results.pose_landmarks.landmark
                shoulder = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                            int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
                elbow = (int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                         int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))

                pixel_dist = np.linalg.norm(np.array(shoulder) - np.array(elbow))
                muac = CALIBRATION_FACTOR * pixel_dist
                st.session_state.user_data["muac"] = muac

                st.image(image, caption="Shoulder to Elbow", use_column_width=True)
                st.success(f"Estimated MUAC: {muac:.2f} cm")

                if st.button("Next"):
                    st.session_state.page = 'summary'
            else:
                st.error("Keypoints not detected. Use a clearer image.")

# --- Step 4: Summary Report ---
def summary():
    data = st.session_state.user_data
    st.title("📋 Summary Report")

    height_m = data["height"] / 100
    bmi = data["weight"] / (height_m ** 2)
    data["bmi"] = bmi

    def bmi_status(b):
        if b < 16:
            return "Severe", "❗", "red"
        elif b < 17:
            return "Moderate", "⚠", "orange"
        elif b < 18.5:
            return "Mild", "yellow"
        else:
            return "Normal", "✅", "green"

    def muac_status(muac):
        if muac < 11.5:
            return "Severe Acute Malnutrition", "❗", "red"
        elif muac < 12.5:
            return "Moderate Acute Malnutrition", "⚠", "orange"
        else:
            return "Normal", "✅", "green"

    bmi_cat, bmi_icon, bmi_color = bmi_status(bmi)
    muac_cat, muac_icon, muac_color = muac_status(data["muac"])

    st.markdown(f"**👶 Name:** {data['name']}")
    st.markdown(f"**🎂 Age:** {data['age']} years")
    st.markdown(f"**⚖️ Weight:** {data['weight']} kg")
    st.markdown(f"**📏 Height:** {data['height']:.2f} cm")
    st.markdown(f"**💪 MUAC:** {data['muac']:.2f} cm")
    st.markdown(f"**📈 BMI:** {bmi:.2f} → *:{bmi_color}[{bmi_cat}]* {bmi_icon}")
    st.markdown(f"**📉 Malnutrition Status:** *:{muac_color}[{muac_cat}]* {muac_icon}")

    if st.button("Back to Main Menu"):
        st.session_state.page = 'main_menu'

# --- Page Router ---
if st.session_state.page == 'login':
    login_screen()
elif st.session_state.page == 'main_menu':
    main_menu()
elif st.session_state.page == 'nutrition_menu':
    nutrition_menu()
elif st.session_state.page == 'input_form':
    input_form()
elif st.session_state.page == 'height_step':
    height_step()
elif st.session_state.page == 'arm_step':
    arm_step()
elif st.session_state.page == 'summary':
    summary()
