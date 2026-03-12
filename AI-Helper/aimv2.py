import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def aimv2(url):
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained(
        "apple/aimv2-large-patch14-224",
        revision="ac764a25c832c7dc5e11871daa588e98e3cdbfb7",
    )
    model = AutoModel.from_pretrained(
        "apple/aimv2-large-patch14-224",
        revision="ac764a25c832c7dc5e11871daa588e98e3cdbfb7",
        trust_remote_code=True,
    )

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

if __name__ == "__main__":
    url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
    outputs = aimv2(url)
    print(outputs)