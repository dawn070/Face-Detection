from ultralytics import YOLO

class Tracker:
    """
    ByteTrack 多目标跟踪器
    封装 Ultralytics 的 model.track() + persist=True，
    实现对图片/视频帧序列中的人脸进行跨帧 ID 跟踪。
    """

    def __init__(self, model_path, conf=0.5, device="cpu"):
        """
        初始化跟踪器
        参数
        ----------
        model_path : str
            YOLO 模型权重文件路径
        conf : float
            检测置信度阈值（低于该值的检测结果会被过滤）
        device : str
            推理设备，可选 "cpu" / "cuda"
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def update(self, frame):
        """
        对当前帧执行检测 + ByteTrack 跟踪
        参数
        ----------
        frame : numpy.ndarray
            OpenCV 读取的 BGR 图像帧
        返回
        ----------
        tracks : list[dict]
            跟踪结果列表，每个元素包含 id / bbox / score / class_name
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            device=self.device,
            persist=True,
            verbose=False
        )

        tracks = []
        result = results[0]

        if result.boxes is None:
            return tracks

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0])
            cls = int(box.cls[0])

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            tracks.append({
                "id": track_id,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "score": score,
                "class_name": self.model.names[cls],
            })

        return tracks