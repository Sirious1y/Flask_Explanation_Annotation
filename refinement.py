import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

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
        mask = np.random.random(trinary_map.shape) < self.p
        trinary_map[mask & (trinary_map == -1)] = 1
        trinary_map[~mask & (trinary_map == -1)] = 0

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

            # Downsample by a factor of 4
            downsampled = cv2.resize(trinary_map_modified, (56, 56), interpolation=cv2.INTER_AREA)

            # Apply a threshold
            _, thresholded = cv2.threshold(downsampled, 0.5, 1, cv2.THRESH_BINARY)

            # Upsample back to original size
            trinary_map_modified = cv2.resize(thresholded, (224, 224), interpolation=cv2.INTER_NEAREST)

            trinary_map_tensor = torch.tensor(trinary_map_modified, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            if isinstance(image, np.ndarray):
                image_tensor = torch.tensor(image, dtype=torch.float32)
            else:
                image_tensor = image

            print(image_tensor.shape)
            print(trinary_map_tensor.shape)
            masked_image = image_tensor * trinary_map_tensor

            # to_pil = transforms.ToPILImage()
            # pil_image = to_pil(masked_image)
            #
            # plt.imshow(pil_image)
            # plt.title('Input')
            # plt.savefig("input.png")
            # plt.close()
            # ori = Image.open("input.png")
            # ori.show()

            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # masked_image = transform(masked_image)

            with torch.no_grad():
                output = self.model(masked_image)
                score = F.softmax(output, dim=1)[0, target_class].item()

            ms.append(trinary_map_modified)
            scores.append(score)

        ms = np.array(ms)
        scores = np.array(scores)
        A = np.sum(ms * scores[:, None, None], axis=0) / (self.num_iterations * self.p)

        min_val = np.min(A)
        max_val = np.max(A)
        print('Normalized')
        A = (A - min_val) / (max_val - min_val)

        A = (A >= 0.2).astype(int)

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
