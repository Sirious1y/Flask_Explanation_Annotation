import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from utils import models_folder, get_labels, process_user_drawing
from refinement import PromptRefinement
from utils import is_front

model_labels = get_labels()

# Function to list available models
def list_models(models_folder):
    return [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f))]

# Function to load the model based on the selected model name
def load_model(model_name):
    model_path = os.path.join(models_folder, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_name}' not found in the 'models' folder.")
    model = torch.load(model_path)
    model.eval()
    return model

def update_labels(model_choice):
    """Fetch and display predefined labels for the chosen model."""
    labels = model_labels.get(model_choice, [])
    return ", ".join(labels)

def update_image_editor(image):
    original_image = Image.fromarray(image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    image = transform(original_image)
    return image, image

def refresh_image_editor(image):
    original_image = Image.fromarray(image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    image = transform(original_image)
    return image

def classify_image(image_editor_important, image_editor_unimportant, model_name):
    if is_front: 
        print("running in the front end")
        return "Sending request to back end. ", None
    else: 
        print("running in back end")

    # Extract images from both editors
    original_image = np.array(image_editor_important['background'])
    important_drawing = np.array(image_editor_important['layers'][0]) if image_editor_important['layers'] else None
    unimportant_drawing = np.array(image_editor_unimportant['layers'][0]) if image_editor_unimportant['layers'] else None

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
    model = load_model(model_name)

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
