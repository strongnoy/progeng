from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import cv2
import numpy as np

app = FastAPI()

class ObjectDetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    def detect(self, img):
        results = self.model(img)
        results.render()

        annotated_img = results.ims[0]
        return annotated_img

detector = ObjectDetector('yolov5s.pt')

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = np.array(pil_img)[:, :, ::-1]

    annotated_img = detector.detect(img)

    result_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))

    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")
