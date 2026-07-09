from camera import Camera
from tracker import Tracker

from utils import (
    FPS,
    draw_boxes,
    draw_fps,
    draw_face_count,
    show_frame
)

import cv2


def main():

    camera = Camera()

    # ---- 使用 ByteTrack 跟踪器 ----
    tracker = Tracker(
        model_path="Models/yolov11n_face_age.pt",
        conf=0.5,
        device="cuda"
    )

    # ---- CPU 模式自动跳帧，GPU 模式全速推理 ----
    # frame_skip = 0 表示每帧都检测
    # frame_skip = N 表示每 N 帧才检测一次（中间帧复用上次结果）
    FRAME_SKIP = 3 if tracker.device == "cpu" else 0

    fps_counter = FPS()

    frame_count = 0
    tracks = []         # 缓存最后一次检测/跟踪结果

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        # 跳帧：FRAME_SKIP=0 时不跳过任何帧
        if frame_count % (FRAME_SKIP + 1) == 0:
            tracks = tracker.update(frame)

        frame_count += 1

        # 计算 FPS
        fps = fps_counter.update()

        # 绘图（tracks 的格式与之前 detections 兼容，
        # 仅在多出 id 字段，draw_boxes 已适配）
        frame = draw_boxes(frame, tracks)
        frame = draw_fps(frame, fps)
        frame = draw_face_count(frame, tracks)

        show_frame("Face Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()