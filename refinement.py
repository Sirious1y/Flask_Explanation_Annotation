import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from utils import *
from explanations import RISE

# Load the pretrained ResNet-50 model
default_model = models.resnet50(pretrained=True)
default_model.eval()  # Set model to evaluation mode
# torch.save(default_model, './models/default.pt')

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

    def get_model_reasoning(self, img, binary_map_include, binary_map_exclude):
        cudnn.benchmark = True
        args = Dummy()

        # Number of workers to load data
        args.workers = 8
        # Size of imput images.
        args.input_size = (224, 224)
        # Size of batches for GPU.
        # Use maximum number that the GPU allows.
        args.gpu_batch = 250

        # Load black box model for explanations
        model = self.model
        model = nn.Sequential(model, nn.Softmax(dim=1))
        model.eval()
        # model = model.cuda()
        for p in model.parameters():
            p.requires_grad = False
        # To use multiple GPUs
        # model = nn.DataParallel(model)

        if binary_map_include is not None:
            binary_map_include = np.ones(args.input_size, dtype=np.float32)
        if binary_map_exclude is not None:
            binary_map_exclude = np.zeros(args.input_size, dtype=np.float32)

        explainer = RISE(model, args.input_size, args.gpu_batch)

        # Generate masks for RISE.
        maskspath = 'masks.npy'
        explainer.generate_masks(N=self.num_iterations, s=8, p1=0.1,
                                 binary_map_include=binary_map_include,
                                 binary_map_exclude=binary_map_exclude,
                                 savepath=maskspath)
        # saliency = explainer(img.cuda()).cpu().numpy()
        saliency = explainer(img).cpu().numpy()
        # p, c = torch.topk(model(img.cuda()), k=1)
        p, c = torch.topk(model(img), k=1)
        p, c = p[0], c[0]
        sal = saliency[c[0]]

        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.axis('off')
        # # plt.title('{:.2f}% {}'.format(100*p[0], get_class_name(c[0])))
        # tensor_imshow(img[0])
        #
        # plt.subplot(1, 2, 2)
        # plt.axis('off')
        # # plt.title(get_class_name(c[0]))
        # tensor_imshow(img[0])
        # plt.imshow(sal, cmap='jet', alpha=0.5)
        # plt.colorbar(fraction=0.046, pad=0.04)
        #
        # plt.savefig("comparison.png")
        # plt.close()
        # processed_drawing = Image.open("comparison.png")
        # processed_drawing.show()

        return sal >= np.quantile(sal, 0.9)

    def refine_prompt(self, image, mask_included, mask_excluded):
        model_mask = self.get_model_reasoning(image, mask_included, mask_excluded)

        # Compute the union of A and B
        A_union_B = np.logical_or(mask_included, model_mask).astype(int)

        # Compute the intersection of (A union B) and C
        intersection = np.logical_and(A_union_B, mask_excluded).astype(int)

        # Compute the final binary mask as A union B minus the intersection
        result_mask = A_union_B - intersection

        return result_mask

    # def refine_prompt(self, image, trinary_map, target_class):
    #     """
    #     Refine the trinary map using the model.
    #
    #     Parameters:
    #     - image: The input image.
    #     - trinary_map: The initial trinary map.
    #
    #     Returns:
    #     - Refined attention map.
    #     """
    #     ms = []
    #     scores = []
    #
    #     for _ in range(self.num_iterations):
    #         trinary_map_modified = self.convert_values(trinary_map.copy())
    #
    #         # Downsample by a factor of 4
    #         downsampled = cv2.resize(trinary_map_modified, (56, 56), interpolation=cv2.INTER_AREA)
    #
    #         # Apply a threshold
    #         _, thresholded = cv2.threshold(downsampled, 0.5, 1, cv2.THRESH_BINARY)
    #
    #         # Upsample back to original size
    #         trinary_map_modified = cv2.resize(thresholded, (224, 224), interpolation=cv2.INTER_NEAREST)
    #
    #         trinary_map_tensor = torch.tensor(trinary_map_modified, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    #
    #         if isinstance(image, np.ndarray):
    #             image_tensor = torch.tensor(image, dtype=torch.float32)
    #         else:
    #             image_tensor = image
    #
    #         masked_image = image_tensor * trinary_map_tensor
    #
    #         transform = transforms.Compose([
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ])
    #         # masked_image = transform(masked_image)
    #
    #         with torch.no_grad():
    #             output = self.model(masked_image)
    #             score = F.softmax(output, dim=1)[0, target_class].item()
    #
    #         ms.append(trinary_map_modified)
    #         scores.append(score)
    #
    #     ms = np.array(ms)
    #     scores = np.array(scores)
    #     A = np.sum(ms * scores[:, None, None], axis=0) / (self.num_iterations * self.p)
    #
    #     min_val = np.min(A)
    #     max_val = np.max(A)
    #     print('Normalized')
    #     A = (A - min_val) / (max_val - min_val)
    #
    #     A = (A >= 0.2).astype(int)
    #
    #     return A
