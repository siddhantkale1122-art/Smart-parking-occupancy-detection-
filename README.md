# Smart Parking Occupancy Detection

A Streamlit computer-vision prototype that estimates whether configured parking spaces are occupied or available. It uses a pretrained YOLOv8 nano model to detect vehicles, then checks overlap between each vehicle bounding box and user-configured parking-space rectangles.

> **Academic project note:** This is a practical baseline implementation, not a guarantee of production-grade accuracy. Measure performance on your own labeled test data and document the results. Do not claim accuracy values that you have not measured.

## Features

- Upload a parking-lot image or a short video.
- Detect cars, motorcycles, buses, and trucks with a pretrained YOLO model.
- Configure parking slots by generating a starter grid or entering rectangle coordinates as JSON.
- Show total, occupied, available, and occupancy percentage.
- Preview slot regions and download an annotated image.
- Validate slot configuration and provide clear input errors.
- Unit tests for overlap logic and invalid slot coordinates.

## How occupancy is estimated

1. YOLO detects vehicle bounding boxes.
2. Each parking slot is represented by a rectangle: `x`, `y`, `w`, and `h`, in pixels of the uploaded image.
3. For each slot, the app calculates the fraction of the slot area intersected by each detected vehicle box.
4. If the largest overlap fraction meets the selected threshold, the slot is marked **Occupied**; otherwise, it is marked **Available**.

This simple overlap method works best when the slot rectangles align with visible parking spaces and the camera view is reasonably clear. Vehicle boxes can overlap adjacent slots, and poor lighting, occlusion, perspective, or distant vehicles can affect results.

## Project structure

```text
SMART-PARKING-FINAL-PROJECT/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── occupancy_detection.py
│   ├── visualization.py
│   └── utils.py
├── tests/
│   └── test_detection.py
├── data/
│   ├── input/
│   ├── annotations/
│   └── processed/
├── models/
│   ├── config/
│   └── weights/
├── outputs/
└── docs/
    ├── PRD_Smart_Parking_Occupancy_Detection.docx
    ├── TRD_Smart_Parking_Occupancy_Detection.docx
    ├── FRD_Smart_Parking_Occupancy_Detection.docx
    └── Synopsis_Smart_Parking_Occupancy_Detection.docx
```

The `docs/` files are the project documentation you supplied. Add them there before pushing to GitHub.

## Run locally on Windows (PowerShell)

Open PowerShell in the project root.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

If PowerShell blocks activation, you do not need to activate the environment when using the full `.venv\Scripts\python.exe` path above.

The first model load may download `yolov8n.pt` from Ultralytics. An internet connection is required for that first download unless the weights are already available locally.

## Slot coordinate format

Coordinates use pixels from the top-left corner of the uploaded image. Example only:

```json
[
  {"id": "A1", "x": 50, "y": 80, "w": 120, "h": 180},
  {"id": "A2", "x": 180, "y": 80, "w": 120, "h": 180}
]
```

Replace these example coordinates with the actual parking-space boundaries in your own image. Coordinates must fit inside the image dimensions, each slot ID must be unique, and width/height must be positive.

The grid option is a convenience for initial testing only; generated rectangles do not automatically know where the real parking spaces are.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Select **Create app**, choose your repository and branch, and set the main file path to `streamlit_app.py`.
4. Deploy. Streamlit installs dependencies from `requirements.txt`.
5. On first use, the YOLO model may need to download; cold starts can take longer and model download must be allowed by the environment.

### Deployment tips

- Start with a short video or image while testing.
- Streamlit Community Cloud resources are limited; the app limits sampled video processing to a configurable maximum.
- If deployment fails during installation, inspect the build logs and adjust dependency versions based on the reported error rather than randomly changing packages.
- Avoid committing `.venv/`, private footage, secrets, or large model weights.
- Only upload footage you have permission to use.
- This app does not provide a live CCTV/camera connection on Streamlit Cloud; uploaded image/video is the supported input.

## Evaluation for the college report

Create a small labeled test set with slot-level ground truth. Compare predicted slot states against the ground truth and report:

- Accuracy = correct slot predictions / all slot predictions
- Precision and recall for the **Occupied** class
- F1-score for the **Occupied** class
- Occupancy count error per image/frame
- Processing time or sampled frames per second

Keep a separate test set that was not used to tune the settings. State the dataset source, number of images/frames, test conditions, and limitations in your final report.

## Mapping to the project documents

- **PRD:** objective, users, scope, success metrics, risks.
- **TRD:** Python, Streamlit, OpenCV, NumPy, Ultralytics YOLO, input/configuration/processing flow.
- **FRD:** image/video input, validation, slot status, aggregate counts, visual feedback, and errors.
- **Synopsis:** problem definition, methodology, expected outcomes, applications, and limitations.

Update the documents if your final implementation differs from this code.
