from ultralyticsplus import YOLO, postprocess_classify_output, postprocess_detect_output
import ultralytics.nn as nn; print('ultralytics.nn', nn);


def yolo(image):
    # load model
    #model = YOLO('keremberke/yolov8m-scene-classification')
    model = YOLO("yolov8l.pt")

    # set model parameters
    model.overrides['conf'] = 0.25  # model confidence threshold

    # perform inference
    results = model.predict(image)

    # observe results
    print(results)
    print("boxes",results[0].boxes)  # Boxes object for bbox outputs
    print("probs", results[0].probs) # [0.1, 0.2, 0.3, 0.4]
    processed_result = postprocess_classify_output(model, result=results[0])
    print(processed_result) # {"cat": 0.4, "dog": 0.6}
    return processed_result

if __name__ == "__main__":
    data = yolo('IMGs/test_cat.jpg')
    outData = []
    for i in data:
        print(i, data[i])
        if data[i] > 0.25:
            outData.append(i)
    print(outData)
