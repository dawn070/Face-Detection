import cv2


class Camera:
    def __init__(self, camera_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头！")

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """
        读取一帧图像
        Returns:
            ret: 是否读取成功
            frame: 图像
        """
        return self.cap.read()

    def release(self):
        """释放摄像头"""
        self.cap.release()

    def get_info(self):
        """获取摄像头信息"""

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        return {
            "width": width,
            "height": height,
            "fps": fps
        }