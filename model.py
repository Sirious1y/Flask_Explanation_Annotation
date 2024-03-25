from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, models
import urllib
from PIL import Image
import os


class ModelClass():
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        if model_path is None:
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, 2)
        else:
            if os.path.isfile(model_path):
                self.model = torch.load(model_path)
            else:
                raise ValueError('Model does not exist')

    def predict(self, input_batch):
        self.model.eval()
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        return probabilities


def string_to_array(text: str):
    string_list = text.strip('[]')

    # Split the string into individual sublist strings
    sublists = string_list.split('], ')

    # Initialize an empty list to store the result
    list_result = []

    # Iterate over each sublist string
    for sublist_str in sublists:
        # Remove any remaining brackets
        sublist_str = sublist_str.strip('[]')
        # Split the sublist string into individual elements
        sublist_elements = sublist_str.split(', ')
        # Convert elements to integers if necessary
        sublist = [int(item) for item in sublist_elements]
        # Append the sublist to the result list
        list_result.append(sublist)

    return np.array(list_result)


def load_image(img_path: str):
    image = Image.open(img_path)
    return image


def process_image(image):
    # Define transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match ResNet input size
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Apply the transformations
    image_tensor = transform(image)
    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Now, image_tensor is ready to be sent into the ResNet model
    # print(image_tensor.shape)  # Should print: torch.Size([1, 3, 224, 224])
    return image_tensor


def masking(image, mask):
    bool_mask = mask.astype(bool)
    image[:, :, ~bool_mask] = 0
    return image

def save_image(image, mask, path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match ResNet input size
    ])

    # Apply the transformations
    image = np.array(transform(image))
    print(image.shape)
    bool_mask = mask.astype(bool)
    image[~bool_mask, :] = 0
    # image = image.squeeze(0)
    print(image.shape)
    image = Image.fromarray(image)
    image.save(path)


def get_result(model_output):
    return torch.argmax(model_output, dim=0).item()

if __name__ == '__main__':
    # img = load_image('./static/images/img_gender/demo_img_1.jpg')
    #
    # ra = np.random.random((3, 3))
    # mask = np.random.randint(2, size=(3, 3))
    #
    # print(mask)
    # print(ra)
    # mask = mask.astype(bool)
    # ra[~mask] = 0
    #
    # print(ra)

    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_ftrs, 2)
    torch.save(resnet18, './model/model1.pt')
