from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from image_detection import ObjectDetector
import torch
import io
from PIL import Image
import cv2

app = FastAPI()
detector = None  # Пока пусто


@app.on_event("startup")
def load_model():
    global detector
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    detector = ObjectDetector(model)


@app.post("/detect/")
async def detect_objects(file: UploadFile):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result_img = detector.detect(img)
    _, encoded_img = cv2.imencode(".jpg", result_img)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
