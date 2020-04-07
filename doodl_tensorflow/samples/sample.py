import numpy as np
from PIL import Image

from doodl import Model, Configuration


if __name__ == '__main__':
    configuration = Configuration(
        {
            "cache": True,
            "endpoint": "http://127.0.0.1:8080/invocations",
            "aws_region": "us-east-1",
            "aws_access_key": "AKIA2PLBLR2YICP2ID2H",
            "aws_secret_access_key": "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",
        }
    )

    image_file = "test.jpg"

    #with open(image_file, "rb") as image:
    #    encoded_string = base64.b64encode(image.read())

    #image = encoded_string.decode("utf-8")

    model = Model(configuration)
    detections = model.inference("s3://vinsa-object-detection/test1.jpg")
    print(detections)

    image = Image.open("test.jpg")
    (width, height) = image.size
    image = np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)

    model = Model(configuration)
    detections = model.inference(image)
    print(detections)

    model = Model(configuration)
    detections = model.inference("http://invalid.jpg")
    print(detections)

