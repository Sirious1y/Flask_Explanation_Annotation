import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import urllib
from PIL import Image
from torchvision import transforms


def save_model():
    resnet18 = models.resnet18(pretrained=True)
    torch.save(resnet18, './model.pt')


if __name__ == '__main__':
    # resnet18 = models.resnet18(pretrained=True)
    # print(resnet18.eval())
    save_model()
    resnet18 = torch.load('./model.pt')

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # num_ftrs = resnet18.fc.in_features
    # resnet18.fc = torch.nn.Linear(num_ftrs, 2)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        resnet18.to('cuda')

    with torch.no_grad():
        output = resnet18(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
