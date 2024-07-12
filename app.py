import gradio as gr
import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import process_user_drawing
from refinement import PromptRefinement

# Path to the models folder
models_folder = 'models'


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


# Function to confirm model and labels
def confirm_labels(model_name, num_labels, label_texts, label_file):
    if label_file is not None:
        label_data = json.load(label_file)
        labels = list(label_data.values())
    else:
        labels = [label.strip() for label in label_texts.split(',')]

    confirmation_message = f"Model: {model_name}\nNumber of Labels: {num_labels}\nLabels: {labels}"
    return confirmation_message, labels


# Function to update image editor with the uploaded image
def update_image_editor(image):
    return image


# Function to classify the image
def classify_image(image_editor_important, image_editor_unimportant, model_name, num_labels, label_texts, label_file):

    # Extract images from both editors
    original_image = np.array(image_editor_important['background'])
    important_drawing = np.array(image_editor_important['layers'][0]) if image_editor_important['layers'] else None
    unimportant_drawing = np.array(image_editor_unimportant['layers'][0]) if image_editor_unimportant['layers'] else None


    # Preprocess the original image for prediction

    if important_drawing is not None:
        important_drawing = Image.fromarray(important_drawing).convert("RGB")
    if unimportant_drawing is not None:
        unimportant_drawing = Image.fromarray(unimportant_drawing).convert("RGB")


    # Process the user's drawings to fill in the circled areas
    image_tensor = torch.tensor(original_image, dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert from HxWxC to CxHxW
    original_image = Image.fromarray(original_image).convert("RGB")
    if important_drawing is not None:
        important_drawing = process_user_drawing(important_drawing)
    if unimportant_drawing is not None:
        unimportant_drawing = process_user_drawing(unimportant_drawing)


    # Initialize a mask with -1 (other areas)
    combined_mask = np.full(important_drawing.shape, -1)

    # Create combined mask: 1 for important, 0 for unimportant, -1 for others
    if important_drawing is not None:
        combined_mask[important_drawing > 0] = 1
    if unimportant_drawing is not None:
        combined_mask[unimportant_drawing > 0] = 0

    # Initialize the prompt refinement class
    prompt_refiner = PromptRefinement(num_iterations=100)

    # Convert the combined mask for the prompt refinement
    refined_mask = prompt_refiner.refine_prompt(image_tensor, combined_mask, target_class=1)  # Example for class 0

    # Display the processed drawing
    plt.imshow(refined_mask, cmap='gray')
    plt.title('Refined')
    plt.savefig("refined_drawing.png")
    plt.close()
    processed_drawing = Image.open("refined_drawing.png")
    processed_drawing.show()

    # Apply the combined mask to the original image
    # Ensure the combined mask is expanded to match the image channels
    print(combined_mask.shape)
    print(refined_mask.shape)
    combined_mask_3d = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.multiply(original_image, combined_mask_3d)
    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))

    masked_image[masked_image > 0] = 255
    masked_image_array = np.abs(masked_image)
    plt.figure(figsize=(6, 6))  # Set the figure size
    plt.imshow(masked_image_array)
    plt.title("Masked Image")
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig("processed_drawing.png")
    plt.close()
    processed_drawing = Image.open("processed_drawing.png")
    processed_drawing.show()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    original_image_tensor = transform(original_image).unsqueeze(0)  # Add batch dimension
    masked_image_tensor = transform(masked_image_pil).unsqueeze(0)  # Add batch dimension

    # Load the model based on the selected model name
    model = load_model(model_name)

    # Make predictions using both original and masked images
    with torch.no_grad():
        outputs_original = model(original_image_tensor)
        outputs_masked = model(masked_image_tensor)

    label_texts = [label.strip() for label in label_texts.split(',')]

    # Get the predicted labels
    predicted_label_original = outputs_original.argmax(dim=1).item()
    predicted_label_masked = outputs_masked.argmax(dim=1).item()

    # Get the label text if available, otherwise default to the label number
    label_text_original = label_texts[predicted_label_original] if label_texts else f"Label {predicted_label_original}"
    label_text_masked = label_texts[predicted_label_masked] if label_texts else f"Label {predicted_label_masked}"

    return f"Original Image Prediction: {label_text_original}\nPrompted Image Prediction: {label_text_masked}"


# Get the list of models
available_models = list_models(models_folder)

# Define Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Image Classification with Custom Labels")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Choose a model and specify labels.")
            model_input = gr.Radio(choices=available_models, label="Choose Model")
            num_labels_input = gr.Textbox(lines=1, placeholder="Enter number of labels", label="Number of Labels")
            label_texts_input = gr.Textbox(lines=2, placeholder="Enter labels separated by commas",
                                           label="Label Texts (optional)")
            label_file_input = gr.File(label="Upload JSON Label File (optional)")
            confirm_button = gr.Button("Confirm Model and Labels")
            confirmation_output = gr.Textbox(label="Confirmation")

        parsed_labels_output = gr.State()

        confirm_button.click(
            confirm_labels,
            inputs=[model_input, num_labels_input, label_texts_input, label_file_input],
            outputs=[confirmation_output, parsed_labels_output]
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 2. Upload and highlight important areas on the image.")
            image_input = gr.Image(label="Upload Image", image_mode='RGB')
            update_button_important = gr.Button("Submit input image")
            image_editor_important = gr.ImageEditor(label="Highlight Important Areas", image_mode='RGB')
            edited_image_important = gr.State()

            update_button_important.click(
                update_image_editor,
                inputs=[image_input],
                outputs=[image_editor_important]
            )

    with gr.Row():
        with gr.Column():
            update_button_unimportant = gr.Button("Submit Indispensable Prompt")
            gr.Markdown("### 3. Label unimportant areas in the image.")
            image_editor_unimportant = gr.ImageEditor(label="Highlight Unimportant Areas", image_mode='RGB')
            edited_image_unimportant = gr.State()

            update_button_unimportant.click(
                update_image_editor,
                inputs=[image_input],
                outputs=[image_editor_unimportant]
            )

    classify_button = gr.Button("Submit Precluded Prompt and Classify Image")

    with gr.Row():
        output_text = gr.Textbox(label="Predicted Labels")

    classify_button.click(
        classify_image,
        inputs=[image_editor_important, image_editor_unimportant, model_input, num_labels_input, label_texts_input, label_file_input],
        outputs=[output_text]
    )

# Launch the interface
demo.launch()

