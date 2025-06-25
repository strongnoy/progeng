import cv2
import torch
from PIL import Image
import io

class ObjectDetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    import io
    from PIL import Image

    def detect(self, img):
        results = self.model(img)
        results.render()

        annotated_img = results.ims[0]
        img_pil = Image.fromarray(annotated_img[:, :, ::-1])  # BGR -> RGB
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer


