import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from insightface.model_zoo import get_model

# =====================================
# 路径配置
# =====================================

FACE_DIR = "Dataset/MyFace/aligned_faces"
LABEL_CSV = "Dataset/MyFace/labels.csv"

DATABASE_DIR = "database"

os.makedirs(DATABASE_DIR, exist_ok=True)

# =====================================
# 加载 ArcFace 模型
# =====================================

model_path = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"

print(model_path)

model = get_model(str(model_path))
model.prepare(ctx_id=-1)

print("ArcFace model loaded.")

# =====================================
# 读取标签
# =====================================

df = pd.read_csv(LABEL_CSV)

embeddings = []

label_dict = {}

# =====================================
# 开始提取
# =====================================

for idx, row in tqdm(df.iterrows(), total=len(df)):

    id_num = row["id"]
    image_name = f"{id_num:06d}"  # 固定6位，不足前面补0

    person_name = row["name"]

    image_path = os.path.join(FACE_DIR, image_name + '.jpg')

    img = cv2.imread(image_path)

    if img is None:

        print("读取失败:", image_name)

        continue

    embedding = model.get_feat(img)

    embedding = embedding.flatten()

    # L2归一化（非常重要）
    embedding /= np.linalg.norm(embedding)

    embeddings.append(embedding)

    label_dict[len(embeddings)-1] = person_name

# =====================================
# 保存数据库
# =====================================

embeddings = np.array(embeddings, dtype=np.float32)

np.save(
    os.path.join(DATABASE_DIR, "embeddings.npy"),
    embeddings
)

with open(
    os.path.join(DATABASE_DIR, "labels.json"),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        label_dict,
        f,
        ensure_ascii=False,
        indent=4
    )

print("="*50)

print("Finished")

print("Embedding shape:", embeddings.shape)

print("Database saved.")