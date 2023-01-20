import cv2
from image import Image


class VehicleDetector:
    """
    Model for detecting vehicles on a photo.
    """

    def __init__(self):
        network = cv2.dnn.readNet(
            "../dnn/yolov4.weights",
            "../dnn/yolov4.cfg",
        )

        self.model = cv2.dnn_DetectionModel(network)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)
        self.classes_allowed = [2, 3, 5, 6, 7]

    def detect_vehicles(self, img: Image):
        vehicles_boxes = []
        class_ids, scores, boxes = self.model.detect(img.img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:
                continue

            if class_id in self.classes_allowed:
                vehicles_boxes.append(box)

        return vehicles_boxes
