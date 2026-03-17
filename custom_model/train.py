from ultralytics import YOLO

model = YOLO("yolo26n.yaml")
results = model.train(data="coco8.yaml", epochs=5)