import cv2


class ObjectDetector:
    def __init__(self, model):
        self.model = model

    def detect(self, img):
        results = self.model(img)
        results.render()

        annotated_img = results.ims[0]  # Изображение с разметкой (в numpy BGR)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return annotated_img

