import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
# 导入对齐工具
from insightface.utils import face_align

# ===========================
# 路径配置
# ===========================
INPUT_DIR = "Dataset/MyFace/images"
OUTPUT_DIR = "Dataset/MyFace/aligned_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG")

# ===========================
# 初始化 InsightFace
# ===========================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)

# ===========================
# 过滤仅图片文件
# ===========================
image_list = []
for fname in sorted(os.listdir(INPUT_DIR)):
    full_path = os.path.join(INPUT_DIR, fname)
    if os.path.isfile(full_path) and fname.endswith(IMG_SUFFIX):
        image_list.append(fname)

print(f"待处理图片总数：{len(image_list)}")
success = 0
failed = 0

# 标准对齐人脸尺寸，ArcFace默认112
ALIGN_SIZE = 112

for img_name in tqdm(image_list):
    img_path = os.path.join(INPUT_DIR, img_name)

    # 二进制读取，解决中文路径cv2.imread返回None
    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取异常 {img_name}：{str(e)}")
        failed += 1
        continue

    if img is None:
        print(f"图片解码失败：{img_name}")
        failed += 1
        continue

    faces = app.get(img)
    if len(faces) == 0:
        print(f"未检测到人脸：{img_name}")
        failed += 1
        continue

    # 取置信度最高人脸
    face = max(faces, key=lambda x: x.det_score)

    # ========== 修复对齐裁剪，替换原来失效的 crop_image ==========
    aligned_face = face_align.norm_crop(img, landmark=face.kps, image_size=ALIGN_SIZE)

    save_path = os.path.join(OUTPUT_DIR, img_name)
    # 使用 imencode + 二进制写入，兼容中文路径
    _, buffer = cv2.imencode(os.path.splitext(img_name)[1] or ".jpg", aligned_face)
    with open(save_path, "wb") as f:
        f.write(buffer.tobytes())
    success += 1

print("=" * 50)
print("处理完成")
print("成功：", success)
print("失败：", failed)