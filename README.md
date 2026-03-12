# AI-image-tagging
school project that tags and identifys content in images

this is a school project that uses a pre-trained model to tag and identify content in images. The project is built using Python and TensorFlow, and it utilizes a convolutional neural network (CNN) to analyze the images and generate tags based on the content. The model is trained on a large dataset of images with corresponding tags, allowing it to learn patterns and associations between visual features and tags. The project includes a user interface where users can upload images and receive tags that describe the content of the image.

file structure:
```
├── README.md
├── requirements.txt
├── AI-helper
│   ├── __init__.py
│   ├── aimv2.py
│   ├── Yolov.py
│   ├── MambaVision_Classification.py
│   ├── MambaVision_FeatureExtraction.py
├── main.py
├── install-venv.sh
```
- `README.md`: This file provides an overview of the project, its purpose, and how to set it up and use it.
- `requirements.txt`: This file lists the Python dependencies required
- `AI-helper/`: This directory contains the helper modules for the AI image tagging project. Each module serves a specific purpose, such as image classification, feature extraction, and object detection.
  - `__init__.py`: This file is used to mark the directory as a Python package.
  - `aimv2.py`: This module contains the main logic for the AI image tagging process, including loading the pre-trained model and generating tags for images.
  - `Yolov.py`: This module implements the YOLO (You Only Look Once) object detection algorithm, which is used to identify and locate objects within images.
  - `MambaVision_Classification.py`: This module is responsible for classifying images based on their content using a pre-trained classification model.
  - `MambaVision_FeatureExtraction.py`: This module extracts features from images, which can be used for various tasks such as image recognition and tagging.
- `main.py`: This is the main entry point of the application. It handles user interactions, such as uploading images and displaying the generated tags.
- `install-venv.sh`: This script is used to set up a virtual environment and install the necessary dependencies for the project. It ensures that all required libraries are installed in an isolated environment, preventing conflicts with other Python projects on the system.