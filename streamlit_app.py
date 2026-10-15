from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.occupancy_detection import OccupancyDetector, detect_slots, validate_slots
from src.visualization import annotate_frame, summary_counts
from src.utils import parse_slots_json, image_to_bgr, safe_filename

st.set_page_config(
    page_title="Smart Parking Occupancy Detection",
    page_icon="🅿️",
    layout="wide",
)

st.title("🅿️ Smart Parking Occupancy Detection")
st.caption("Computer vision for estimating vacant and occupied parking spaces.")
st.info(
    "This project uses a pretrained YOLO model to detect vehicles, then compares vehicle "
    "bounding boxes with parking-space rectangles. Results depend on camera angle, lighting, "
    "occlusion, slot coordinates, and model confidence."
)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Vehicle detection confidence", 0.10, 0.90, 0.35, 0.05)
    overlap_threshold = st.slider("Slot overlap threshold", 0.05, 0.90, 0.20, 0.05)
    st.caption("A slot is occupied when a detected vehicle overlaps it by at least this fraction.")
    st.divider()
    st.markdown("**Vehicle classes**")
    st.write("Car, motorcycle, bus, truck")
    st.caption("The first run downloads the small pretrained YOLOv8 model.")

@st.cache_resource(show_spinner="Loading vehicle detection model…")
def load_detector():
    return OccupancyDetector()

def make_default_slots(width: int, height: int, rows: int, cols: int, margin_pct: float):
    margin_x = int(width * margin_pct / 100.0)
    margin_y = int(height * margin_pct / 100.0)
    usable_w = max(1, width - 2 * margin_x)
    usable_h = max(1, height - 2 * margin_y)
    cell_w = usable_w / cols
    cell_h = usable_h / rows
    result = []
    for r in range(rows):
        for c in range(cols):
            result.append({
                "id": f"{chr(65 + r)}{c + 1}",
                "x": int(margin_x + c * cell_w),
                "y": int(margin_y + r * cell_h),
                "w": max(1, int(cell_w)),
                "h": max(1, int(cell_h)),
            })
    return result

def process_image(image_bgr, slots, detector, conf, overlap):
    vehicles = detector.detect(image_bgr, confidence=conf)
    statuses = detect_slots(slots, vehicles, image_bgr.shape[1], image_bgr.shape[0],
                            overlap_threshold=overlap)
    annotated = annotate_frame(image_bgr.copy(), slots, vehicles, statuses)
    return annotated, vehicles, statuses

mode = st.radio("Choose input", ["Image", "Video"], horizontal=True)
uploaded = st.file_uploader(
    "Upload a parking-lot image" if mode == "Image" else "Upload a parking-lot video",
    type=["jpg", "jpeg", "png"] if mode == "Image" else ["mp4", "mov", "avi", "m4v"],
)

if uploaded is None:
    st.subheader("How to use")
    st.markdown(
        "1. Upload a clear parking-lot image or short video.\n"
        "2. Configure parking-slot rectangles. You can generate a starter grid or paste JSON coordinates.\n"
        "3. Run detection and review the occupied/vacant counts.\n\n"
        "**Important:** Generated grid slots are only a starting point. Adjust them to fit the actual parking spaces."
    )
    st.stop()

# Read the first frame so slot coordinates can be configured against its dimensions.
if mode == "Image":
    try:
        pil_image = Image.open(uploaded).convert("RGB")
        frame_bgr = image_to_bgr(pil_image)
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
        st.stop()
else:
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
        temp_in.write(uploaded.getbuffer())
        video_path = temp_in.name
    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr = cap.read()
    cap.release()
    try:
        os.unlink(video_path)
    except OSError:
        pass
    if not ok or frame_bgr is None:
        st.error("Could not read the uploaded video. Try an MP4 encoded with H.264.")
        st.stop()

height, width = frame_bgr.shape[:2]
st.caption(f"Input dimensions: {width} × {height} pixels")

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Configure parking slots")
    config_mode = st.radio("Slot setup", ["Generate grid", "Enter slot JSON"], horizontal=True)
    if config_mode == "Generate grid":
        c1, c2, c3 = st.columns(3)
        with c1:
            rows = st.number_input("Rows", min_value=1, max_value=12, value=2, step=1)
        with c2:
            cols = st.number_input("Slots per row", min_value=1, max_value=20, value=5, step=1)
        with c3:
            margin_pct = st.number_input("Margin (%)", min_value=0, max_value=30, value=5, step=1)
        slots = make_default_slots(width, height, int(rows), int(cols), float(margin_pct))
        st.warning("A grid is not a parking-space detector. Review the overlay and use JSON coordinates for real slot boundaries.")
        slots_json = json.dumps(slots, indent=2)
        with st.expander("Edit generated slot JSON"):
            slots_json = st.text_area("Slot rectangles (pixels)", value=slots_json, height=260)
    else:
        example = [
            {"id": "A1", "x": 50, "y": 80, "w": 120, "h": 180},
            {"id": "A2", "x": 180, "y": 80, "w": 120, "h": 180},
        ]
        slots_json = st.text_area("Slot rectangles (pixels)", value=json.dumps(example, indent=2), height=260)
        st.caption("Replace example coordinates with your image's pixel coordinates. Each rectangle needs id, x, y, w, h.")
    try:
        slots = parse_slots_json(slots_json)
        validate_slots(slots, width, height)
        st.success(f"{len(slots)} parking slot(s) configured.")
    except (ValueError, TypeError) as exc:
        st.error(f"Slot configuration error: {exc}")
        st.stop()

with right:
    st.subheader("2. Preview slot regions")
    preview = annotate_frame(frame_bgr.copy(), slots, [], {})
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption("Green outlines show configured slot regions before vehicle detection.")

if mode == "Video":
    st.subheader("Video processing limits")
    video_col1, video_col2 = st.columns(2)
    with video_col1:
        sample_every = st.number_input("Process every Nth frame", min_value=1, max_value=30, value=3, step=1)
    with video_col2:
        max_frames = st.number_input("Maximum sampled frames", min_value=10, max_value=600, value=120, step=10)

st.divider()
run = st.button("🚗 Run occupancy detection", type="primary", use_container_width=True)
if run:
    try:
        detector = load_detector()
        if mode == "Image":
            with st.spinner("Detecting vehicles and estimating occupancy…"):
                annotated, vehicles, statuses = process_image(
                    frame_bgr, slots, detector, confidence, overlap_threshold
                )
            counts = summary_counts(statuses)
            a, b, c, d = st.columns(4)
            a.metric("Total slots", counts["total"])
            b.metric("Occupied", counts["occupied"])
            c.metric("Available", counts["available"])
            d.metric("Occupancy", f'{counts["occupancy_pct"]:.1f}%')
            st.subheader("Detection result")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.download_button(
                "Download annotated image",
                data=cv2.imencode(".jpg", annotated)[1].tobytes(),
                file_name="parking_occupancy_result.jpg",
                mime="image/jpeg",
            )
            st.subheader("Slot-level results")
            st.dataframe(statuses, use_container_width=True, hide_index=True)
            with st.expander("Detected vehicles"):
                st.write(f"Vehicle detections: {len(vehicles)}")
                st.json(vehicles)
        else:
            # Re-open uploaded video and process every Nth frame to control runtime.
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
                temp_in.write(uploaded.getbuffer())
                video_path = temp_in.name
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 20.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            output_frames = []
            last_statuses = []
            frame_index = 0
            processed = 0
            progress = st.progress(0)
            status_text = st.empty()
            with st.spinner("Processing video…"):
                while cap.isOpened() and processed < int(max_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % int(sample_every) == 0:
                        vehicles = detector.detect(frame, confidence=confidence)
                        statuses = detect_slots(slots, vehicles, frame.shape[1], frame.shape[0],
                                                overlap_threshold=overlap_threshold)
                        annotated = annotate_frame(frame, slots, vehicles, statuses)
                        output_frames.append(annotated)
                        last_statuses = statuses
                        processed += 1
                        progress.progress(min(1.0, processed / int(max_frames)))
                        status_text.write(f"Processed sampled frame {processed}")
                    frame_index += 1
                cap.release()
            try:
                os.unlink(video_path)
            except OSError:
                pass
            if not output_frames:
                st.error("No video frames could be processed.")
                st.stop()
            counts = summary_counts(last_statuses)
            a, b, c, d = st.columns(4)
            a.metric("Total slots", counts["total"])
            b.metric("Occupied (last sampled frame)", counts["occupied"])
            c.metric("Available (last sampled frame)", counts["available"])
            d.metric("Occupancy (last sampled frame)", f'{counts["occupancy_pct"]:.1f}%')
            st.subheader("Last processed frame")
            st.image(cv2.cvtColor(output_frames[-1], cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Processed {processed} sampled frames out of {frame_index} decoded frames. Counts above refer to the last sampled frame.")
            st.dataframe(last_statuses, use_container_width=True, hide_index=True)
            st.warning("Video output export is not included in this deployment-friendly starter version; the app reports and displays the last sampled frame.")
    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)
