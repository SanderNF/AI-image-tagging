from ultralyticsplus import YOLO, postprocess_classify_output


def yolo(image):
    # load model
    model = YOLO('keremberke/yolov8m-scene-classification')

    # set model parameters
    model.overrides['conf'] = 0.25  # model confidence threshold

    # perform inference
    results = model.predict(image)

    # observe results
    print(results[0].probs) # [0.1, 0.2, 0.3, 0.4]
    processed_result = postprocess_classify_output(model, result=results[0])
    print(processed_result) # {"cat": 0.4, "dog": 0.6}

if __name__ == "__main__":
    yolo('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg')
