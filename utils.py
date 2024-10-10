import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from PIL import Image
from config import model_folder

def get_labels(): 
    # Dictionary mapping models to their predefined labels
    default_labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    default_labels = [label.split(' ', 1)[1].split(',')[0] for label in default_labels]
    # print(default_labels)
    model_labels = {
        "resnet50.pt": default_labels,
        "vgg16.pt": ["labelA", "labelB"]
        # Add more models and their labels as needed
    }

    return model_labels

# Function to process the user's drawing to fill in the circled areas
def process_user_drawing(user_drawing):
    user_drawing = np.array(user_drawing)  # Convert to numpy array
    user_drawing_gray = cv2.cvtColor(user_drawing, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(user_drawing_gray, 1, 255, cv2.THRESH_BINARY)  # Threshold the drawing
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    cv2.drawContours(user_drawing_gray, contours, -1, (255), thickness=cv2.FILLED)  # Fill the contours
    return user_drawing_gray

# Function to list available models
def list_models(models_folder):
    return [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f))]

# Function to load the model based on the selected model name
def load_model(model_name):
    model_path = os.path.join(model_folder, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_name}' not found in the 'models' folder.")
    model = torch.load(model_path)
    model.eval()
    return model

# Dummy class to store arguments
class Dummy():
    pass

# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])

# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])


# Image preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalization for ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)
