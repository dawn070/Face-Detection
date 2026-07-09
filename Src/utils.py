import cv2
import time


# ----------------------------
# FPS计算器
# ----------------------------
class FPS:
    def __init__(self):
        self.prev_time = time.time()

    def update(self):
        """
        返回当前FPS
        """
        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)

        self.prev_time = current_time

        return fps


# ----------------------------
# 绘制检测框（支持 Track ID）
# ----------------------------
def draw_boxes(frame, detections):
    """
    在帧上绘制检测框和标签。

    标签格式：
        - 有跟踪 ID：ID:1 98%
        - 无跟踪 ID：Face 98%  （首帧或跟踪丢失时的回退）
    """

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        score = det["score"]

        # --- 构造标签文本 ------------------------------------------------
        track_id = det.get("id")
        age = det.get("class_name", "")
        if track_id is not None:
            # ByteTrack 分配了稳定 ID，附带年龄段
            label = f"ID:{track_id} {age} {score:.0%}"
        else:
            # 首帧或跟踪丢失，回退到年龄段
            label = f"{age} {score:.0%}"

        # 画矩形框
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # --- 动态计算标签背景宽度 -----------------------------------------
        (label_width, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_bg_x2 = x1 + label_width + 10   # 左右各留 5px 边距

        # 标签背景
        cv2.rectangle(
            frame,
            (x1, y1 - 25),
            (label_bg_x2, y1),
            (0, 255, 0),
            -1
        )

        # 标签文字
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame


# ----------------------------
# 绘制FPS
# ----------------------------
def draw_fps(frame, fps):

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return frame


# ----------------------------
# 显示人脸数量
# ----------------------------
def draw_face_count(frame, detections):

    count = len(detections)

    cv2.putText(
        frame,
        f"Face Count: {count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    return frame


# ----------------------------
# 显示窗口
# ----------------------------
def show_frame(window_name, frame):

    cv2.imshow(window_name, frame)