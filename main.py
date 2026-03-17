import os, AI_Helper
image_folder = "./IMGs"
image_list = os.listdir(image_folder)
print(image_list)
for image in image_list:
    image_path = os.path.join(image_folder, image)
    detections = AI_Helper.Yolov.run_yolo(image=image_path)
    print(f"Detections for {image}:")
    for det in detections:
        print(f"  Detected {det['class_name']} with confidence {det['confidence']:.2f}")