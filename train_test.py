from ultralytics import YOLO

if __name__ == '__main__':
    # 加载基础预训练模型
    model = YOLO("yolov8n.pt")

    # 开始训练
    model.train(
        data="coco8.yaml",
        epochs=10,
        batch=8,
        imgsz=960,
        workers=0,
        device=0,
        patience=3
    )