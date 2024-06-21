import gradio as gr
import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

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
def classify_image(image_editor, model_name, num_labels, label_texts, label_file):

    print(f'layers[0]: {image_editor['layers'][0]}')
    print(f'background: {image_editor['background']}')
    # Separate the input into the original image and the user's drawing
    original_image = np.array(image_editor['background'])
    # user_drawing = np.array(image_editor['composite'])
    user_drawing = np.array(image_editor['layers'][0]) if image_editor['layers'] else None

    # Preprocess the original image for prediction
    original_image = Image.fromarray(original_image)
    user_drawing = Image.fromarray(user_drawing)
    # original_image.show()
    user_drawing.show()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    original_image = transform(original_image).unsqueeze(0)  # Add batch dimension

    # Load the model based on the selected model name
    model = load_model(model_name)

    # Make prediction using the original image
    with torch.no_grad():
        outputs = model(original_image)

    # Get the predicted label
    predicted_label = outputs.argmax(dim=1).item()
    label_texts = [label.strip() for label in label_texts.split(',')]

    return label_texts[predicted_label] if label_texts else f"Label {predicted_label}"


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
            update_button = gr.Button("Edit Image")
            image_editor = gr.ImageEditor(label="Highlight Image", image_mode='RGB')
            edited_image = gr.State()

        update_button.click(
            update_image_editor,
            inputs=[image_input],
            outputs=[image_editor]
        )

    classify_button = gr.Button("Classify Image")

    with gr.Row():
        output_text = gr.Textbox(label="Predicted Labels")

    classify_button.click(
        classify_image,
        inputs=[image_editor, model_input, num_labels_input, label_texts_input, label_file_input],
        outputs=[output_text]
    )

# Launch the interface
demo.launch()
