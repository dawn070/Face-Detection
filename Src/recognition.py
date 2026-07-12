import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont

# ============================
# 配置
# ============================

IMAGE_PATH = r"C:\Users\23978\Downloads\mmexport1609516008416.jpg"      # 测试图片
DATABASE = "Dataset/MyFace/database/embeddings.npy"  # 数据库路径
LABELS = "Dataset/MyFace/database/labels.json"

THRESHOLD = 0.40

# ============================
# 加载数据库
# ============================

database = np.load(DATABASE)

with open(LABELS, "r", encoding="utf-8") as f:
    labels = json.load(f)

# ============================
# 初始化 InsightFace
# ============================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]   # GPU改CUDAExecutionProvider
)

app.prepare(
    ctx_id=-1,
    det_size=(640,640)
)

# ============================
# 读取图片
# ============================

img = cv2.imread(IMAGE_PATH)

faces = app.get(img)

print(f"检测到 {len(faces)} 张人脸")

# ============================
# 识别
# ============================

for face in faces:

    embedding = face.embedding.astype(np.float32)

    embedding /= np.linalg.norm(embedding)

    similarity = database @ embedding

    best_idx = np.argmax(similarity)

    best_score = similarity[best_idx]

    if best_score > THRESHOLD:
        name = labels[str(best_idx)]
    else:
        name = "Unknown"

    box = face.bbox.astype(int)

    x1, y1, x2, y2 = box
    box_w = x2 - x1

    # 红框
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0,0,255),
        2
    )

    text = f"{name} {best_score:.2f}"

    # 使用 PIL 绘制中文（cv2.putText 不支持中文）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows 黑体
    font_size = max(12, min(48, int(box_w * 0.08)))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    draw.text((x1, y1 - font_size - 8), text, font=font, fill=(0, 0, 255))

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ============================
# 保存结果
# ============================

cv2.imwrite("result.jpg", img)

print("识别完成")