from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
import shutil
import os
import uuid
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

@app.post("/detect")
async def detect_pest(image: UploadFile = File(...)):
    import tempfile

    # Create temp file path (works on Windows / Linux / Mac)
    temp_name = f"{uuid.uuid4()}.jpg"
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, temp_name)

    # Save file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    print("Saved file:", temp_path, "size:", os.path.getsize(temp_path))

    # YOLO prediction
    results = model.predict(temp_path, imgsz=640, conf=0.05)
    boxes = results[0].boxes

    # No detection
    if boxes is None or len(boxes) == 0:
        os.remove(temp_path)
        return {"pestType": None}

    # Get highest confidence detection
    ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    best = confs.argmax()

    pest_id = ids[best]
    confidence = float(confs[best])
    name = model.names[pest_id]

    os.remove(temp_path)

    return {
        "pestType": name,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
