from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once at startup
model = YOLO("best.pt")

@app.post("/detect/")
async def detect_pest(image: UploadFile = File(...)):
    # Read uploaded image bytes
    img_bytes = await image.read()

    # Convert to numpy array
    np_arr = np.frombuffer(img_bytes, np.uint8)

    # Decode image from bytes
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    # YOLO prediction directly on decoded image
    results = model.predict(img, imgsz=640, conf=0.05)
    boxes = results[0].boxes

    # No detections
    if boxes is None or len(boxes) == 0:
        return {"pestType": None}

    # Get highest confidence detection
    ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    best_idx = confs.argmax()

    pest_name = model.names[ids[best_idx]]
    confidence = float(confs[best_idx])

    return {
        "pestType": pest_name,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
