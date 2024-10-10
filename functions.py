import os
import requests
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from utils import get_labels, process_user_drawing
from refinement import PromptRefinement
# from config import is_front, API_URL

model_labels = get_labels()

def classify_image(original_image, important_drawing, unimportant_drawing, model_name, model):
    # Preprocess the original image for prediction
    original_image = Image.fromarray(original_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        lambda x: torch.unsqueeze(x, 0)
    ])
    image_tensor = transform(original_image)

    # Process the user's drawings to fill in the circled areas
    if important_drawing is not None:
        important_drawing = Image.fromarray(important_drawing).convert("RGB")
        important_drawing = process_user_drawing(important_drawing)
    if unimportant_drawing is not None:
        unimportant_drawing = Image.fromarray(unimportant_drawing).convert("RGB")
        unimportant_drawing = process_user_drawing(unimportant_drawing)

    # Initialize a mask with -1 (other areas)
    combined_mask = np.full(important_drawing.shape, -1)
    if important_drawing is not None:
        combined_mask[important_drawing > 0] = 1
    if unimportant_drawing is not None:
        combined_mask[unimportant_drawing > 0] = 0

    # Load the entire model
    # model = load_model(model_name)

    # Initialize the prompt refinement class
    prompt_refiner = PromptRefinement(model=model, num_iterations=50)
    refined_mask = prompt_refiner.refine_prompt(image_tensor, important_drawing, unimportant_drawing)
    # # Display the processed drawing
    # plt.imshow(refined_mask, cmap='gray')
    # plt.title('Refined')
    # plt.savefig("refined_drawing.png")
    # plt.close()
    # processed_drawing = Image.open("refined_drawing.png")
    # processed_drawing.show()

    # Apply the combined mask to the original image
    combined_mask_3d = np.repeat(refined_mask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.multiply(original_image, combined_mask_3d)
    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))

    # Display the processed drawing
    # plt.imshow(masked_image)
    # plt.title('Masked')
    # plt.savefig("masked.png")
    # plt.close()
    # masked = Image.open("masked.png")
    # masked.show()

    # original_image_tensor = transform(original_image).unsqueeze(0)
    masked_image_tensor = transform(masked_image_pil)

    # model = model.to(torch.device('cpu'))

    with torch.no_grad():
        outputs_original = model(image_tensor)
        outputs_masked = model(masked_image_tensor)

    predicted_label_original = outputs_original.argmax(dim=1).item()
    predicted_label_masked = outputs_masked.argmax(dim=1).item()

    label_texts = model_labels.get(model_name, [])
    label_text_original = label_texts[predicted_label_original] if label_texts else f"Label {predicted_label_original}"
    label_text_masked = label_texts[predicted_label_masked] if label_texts else f"Label {predicted_label_masked}"

    return f"Original Image Prediction: {label_text_original}\nPrompted Image Prediction: {label_text_masked}", masked_image
