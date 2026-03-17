import requests
from PIL import Image
from transformers import AutoImageProcessor, FlaxAutoModel, AutoModel



def aimv2(url):
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained(
        "apple/aimv2-1B-patch14-448",
        revision="7f292735d3a07a911559c0fabb3ad3e9d141713f",
    )
    model = AutoModel.from_pretrained(
        "apple/aimv2-1B-patch14-448",
        revision="7f292735d3a07a911559c0fabb3ad3e9d141713f",
        trust_remote_code=True,
    )

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return outputs, embedding.detach().cpu().numpy()

if __name__ == "__main__":
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    outputs = aimv2(url)
    print("returned", outputs)