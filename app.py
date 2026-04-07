import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO

st.set_page_config(layout="wide")

st.title("🚦 AI Smart Traffic Control System")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Upload videos
videos = st.file_uploader(
    "Upload 4 lane videos",
    type=["mp4"],
    accept_multiple_files=True
)

if videos and len(videos) == 4:

    st.success("Videos loaded successfully")

    # Save videos temporarily
    temp_files = []
    for vid in videos:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid.read())
        temp_files.append(tfile.name)

    caps = [cv2.VideoCapture(p) for p in temp_files]

    frame_placeholders = [st.empty() for _ in range(4)]

    current_lane = 0
    state = "GREEN"
    last_switch = time.time()

    while True:
        frames = []

        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                st.warning("Video ended")
                st.stop()

            # YOLO detection
            results = model(frame)
            annotated = results[0].plot()

            frames.append(annotated)

        # Display 2x2 grid
        col1, col2 = st.columns(2)
        col1.image(frames[0], channels="BGR")
        col1.image(frames[1], channels="BGR")
        col2.image(frames[2], channels="BGR")
        col2.image(frames[3], channels="BGR")

        # Traffic logic
        elapsed = time.time() - last_switch

        if state == "GREEN":
            if elapsed > 10:
                state = "YELLOW"
                last_switch = time.time()

        elif state == "YELLOW":
            if elapsed > 3:
                current_lane = (current_lane + 1) % 4
                state = "GREEN"
                last_switch = time.time()

        st.write(f"Current Lane: {current_lane+1} | State: {state}")

        time.sleep(0.03)
