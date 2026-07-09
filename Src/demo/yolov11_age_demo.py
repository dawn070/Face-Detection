from ultralytics import YOLO

model_path = 'Models/yolov11n_face_age.pt'
# 加载预训练人脸年龄模型 yolo11n-face-age
model = YOLO(model_path)
print(model)

# 测试图片路径
source = 'Src/demo/image/zrn.jpg'

# 推理，save=True 保存检测结果图片
results = model.predict(source=source, save=True)

# 正确读取人脸+年龄区间
for res in results:
    name_map = res.names
    face_boxes = res.boxes
    if face_boxes:
        class_idx = face_boxes.cls.cpu().numpy().astype(int)
        confidence = face_boxes.conf.cpu().numpy()
        coords = face_boxes.xyxy.cpu().numpy()
        
        for idx, conf, pos in zip(class_idx, confidence, coords):
            age_label = name_map[idx]
            print(f"人脸预测年龄段：{age_label}，置信度：{conf:.2f}")