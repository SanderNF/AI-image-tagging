import os, AI_Helper

def load_tags(tags_file='tags.json'):
    import json
    with open(tags_file, 'r') as f:
        tags = json.load(f)["yolov8"]
    return tags

def in_tags(key, tags):
    try:
        return tags[key]    
    except KeyError:
        print(f"Key {key} not found in tags. Returning None.")
        return None
    
def threshold_checker(detections, fallback_threshold=0.5):
    output = {"good": [], "bad": []}
    tags = load_tags()
    for det in detections:
        print(f"Checking detection: {det['class_name']} (ID: {det['class_id']}) with confidence {det['confidence']:.2f}")
        tag = in_tags(str(det['class_id']), tags)
        print(f"Tag for class_id {det['class_id']}: {tag}")
        if tag and det['confidence'] > tag['confidence_threshold']:
            output['good'].append(det)
            print(f"  -> GOOD: {det['class_name']} (ID: {det['class_id']}) with confidence {det['confidence']:.2f} exceeds threshold {tag['confidence_threshold']:.2f}")
        elif tag == None and det['confidence'] > fallback_threshold:
            output['good'].append(det)
            print(f"  -> GOOD (fallback): {det['class_name']} (ID: {det['class_id']}) with confidence {det['confidence']:.2f} exceeds fallback threshold {fallback_threshold:.2f}")
        else:
            output['bad'].append(det)
            print(f"  -> BAD: {det['class_name']} (ID: {det['class_id']}) with confidence {det['confidence']:.2f} does not exceed fallback threshold {fallback_threshold:.2f}")
    return output

def main():
    print("Running YOLOv8 on test images...")
    print(load_tags())
    print(load_tags()["0"]['confidence_threshold'])
    image_folder = "./IMGs"
    image_list = os.listdir(image_folder)
    print(image_list)
    for image in image_list:
        image_path = os.path.join(image_folder, image)
        detections = AI_Helper.Yolov.run_yolo(image=image_path)
        print(f"Detections for {image}:")
        results = threshold_checker(detections)
        print("valid detections:")
        for det in results['good']:
            print(f"  Detected {det['class_name']} ({det['class_id']}) with confidence {det['confidence']:.2f}")
        print("invalid detections:")
        for det in results['bad']:
            print(f"  Detected {det['class_name']} ({det['class_id']}) with confidence {det['confidence']:.2f}")


if __name__ == "__main__":
    main()