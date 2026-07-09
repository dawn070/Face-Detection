from ultralytics import YOLO


class Detector:
    """
    YOLOv8 Face Detector
    """

    def __init__(
            self,
            model_path="models/face.pt",
            conf=0.5,
            device="cpu"
    ):

        # 加载模型
        self.model = YOLO(model_path)

        self.conf = conf
        self.device = device

    def detect(self, frame):
        """
        参数
        ----------
        frame : ndarray(OpenCV BGR)

        返回
        ----------
        detections : list
        """

        results = self.model(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        detections = []

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            score = float(box.conf[0])

            cls = int(box.cls[0])

            detections.append({

                "bbox": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ),

                "score": score,

                "class_id": cls,

                "class_name": self.model.names[cls]

            })

        return detections