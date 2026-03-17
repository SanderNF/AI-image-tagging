from ultralytics import YOLO
import json
#yolov8cls = YOLO("yolov8l.pt", task="classify")

#model = yolov8cls
#source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#result = model.predict(source=source, save=True, save_txt=True, save_conf=True, save_crop=True, conf=0.25, verbose=True)
#print(result)
def run_yolo(yolo_model="yolov8x.pt",image='IMGs/test_cat.jpg'):
    model = YOLO(yolo_model)
    results = model(image, save=True)  # or model(img), model([imgs])

    # results is a list; each item corresponds to one input image
    r = results[0]

    # Per-detection tensors
    boxes = r.boxes  # ultralytics.engine.results.Boxes object

    # Get numpy arrays (N x 4 for xyxy, N for conf, N for class)
    xyxy = boxes.xyxy.cpu().numpy()     # [[x1,y1,x2,y2], ...]
    conf = boxes.conf.cpu().numpy()     # [0.86, 0.75, ...]
    cls  = boxes.cls.cpu().numpy().astype(int)  # [15, 57, ...]

    # Class id → name mapping
    names = r.names  # {0: "person", 1: "bicycle", ...}

    # Build a parsed list
    detections = []
    for xy, c, cls_id in zip(xyxy, conf, cls):
        detections.append({
            "xyxy": xy.tolist(),
            "confidence": float(c),
            "class_id": int(cls_id),
            "class_name": names[int(cls_id)],
        })
    return detections

if __name__ == "__main__":
    detections = run_yolo()
    print(detections)
    with open('detections.json', 'w') as f:
        json.dump(detections, f, indent=4)

    for det in detections:
        print(f"Detected {det['class_name']} with confidence {det['confidence']:.2f}")