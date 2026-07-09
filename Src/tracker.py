"""
目标跟踪模块 —— 基于 Ultralytics 内置 ByteTrack

用法：
    tracker = Tracker(model_path="Models/yolov8n_face.pt")
    tracks = tracker.update(frame)

返回格式：
    [
        {
            "id": 1,                    # 跟踪 ID（ByteTrack 分配）
            "bbox": (x1, y1, x2, y2),   # 边界框坐标
            "score": 0.98,              # 置信度
            "class_name": "face"        # 类别名称
        },
        ...
    ]

后续可在此模块扩展：
    - 轨迹绘制（draw_trails）
    - 停留时间统计（dwell_time）
    - ROI 区域检测（in_roi）
"""

from ultralytics import YOLO


class Tracker:
    """
    ByteTrack 多目标跟踪器

    封装 Ultralytics 的 model.track() + persist=True，
    实现对视频帧序列中的人脸进行跨帧 ID 跟踪。
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
        # 加载 YOLO 模型（内部自带 ByteTrack 跟踪头）
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
        # --- 使用 model.track() 启用 ByteTrack ---------------------------
        # persist=True 表示跨帧维持 ID 状态，不可省略
        results = self.model.track(
            frame,
            conf=self.conf,
            device=self.device,
            persist=True,
            verbose=False
        )

        tracks = []

        result = results[0]

        # 没有检测到任何目标时直接返回空列表
        if result.boxes is None:
            return tracks

        for box in result.boxes:
            # 边界框坐标（左上角 + 右下角）
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # 置信度
            score = float(box.conf[0])

            # 类别索引
            cls = int(box.cls[0])

            # --- 提取 ByteTrack 分配的跟踪 ID ----------------------------
            # box.id 为 None 说明跟踪器尚未初始化（首帧或跟踪丢失）
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
