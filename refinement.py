import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Load the pretrained ResNet-50 model
default_model = models.resnet50(pretrained=True)
default_model.eval()  # Set model to evaluation mode


class PromptRefinement:
    def __init__(self, model=default_model, num_iterations=1000, p=0.2):
        """
        Initialize the Visual Attention Prompt Refinement module.

        Parameters:
        - model: The pre-trained model used for generating predictions.
        - num_iterations: Number of perturbations.
        - p: Probability to set undecided pixels to indispensable.
        """
        self.model = model
        self.num_iterations = num_iterations
        self.p = p

    def convert_values(self, trinary_map):
        # Create a mask for -1 values based on probability p
        mask_minus1 = np.random.random(trinary_map.shape) < self.p
        trinary_map[mask_minus1 & (trinary_map == -1)] = 0

        # Create a mask for 1 values based on probability 1-p
        mask_1 = np.random.random(trinary_map.shape) < (1 - self.p)
        trinary_map[mask_1 & (trinary_map == 1)] = 0

        return trinary_map

    def refine_prompt(self, image, trinary_map, target_class):
        """
        Refine the trinary map using the model.

        Parameters:
        - image: The input image.
        - trinary_map: The initial trinary map.

        Returns:
        - Refined attention map.
        """
        ms = []
        scores = []

        for _ in range(self.num_iterations):
            trinary_map_modified = self.convert_values(trinary_map.copy())
            trinary_map_tensor = torch.tensor(trinary_map_modified, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            if isinstance(image, np.ndarray):
                image_tensor = torch.tensor(image, dtype=torch.float32)
            else:
                image_tensor = image

            masked_image = image_tensor * trinary_map_tensor

            with torch.no_grad():
                output = self.model(masked_image)
                score = F.softmax(output, dim=1)[0, target_class].item()

            ms.append(trinary_map_modified)
            scores.append(score)

        ms = np.array(ms)
        scores = np.array(scores)
        A = np.sum(ms * scores[:, None, None], axis=0) / (self.num_iterations * self.p)
        return A


# for i in range(num_iterations=1000):
#     # Generate weighted perturbation map
#     trinary_map_modified = convert_values(trinary_map, p) # one M_i
#     score = resnet(trinary_map_modified \cdoc image) # the label of resnet(image)
#
#     ms.append(trinary_map_modified)
#     scores.append(score)
#
# A = ms * scores / (num_iterations * p)
