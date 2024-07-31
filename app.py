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

# Dictionary mapping models to their predefined labels
model_labels = {
    "default.pt": ["Male", "Female"],
    "resnet50.pt": ["Male", "Female"],
    "vgg16.pt": ["labelA", "labelB"]
    # Add more models and their labels as needed
}

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
    print('Image Received')
    return {
        image_editor_important: image,
        image_editor_unimportant: image
    }

def classify_image(image_editor_important, image_editor_unimportant, model_name):
    original_image = np.array(image_editor_important['background'])
    important_drawing = np.array(image_editor_important['layers'][0]) if image_editor_important['layers'] else None
    unimportant_drawing = np.array(image_editor_unimportant['layers'][0]) if image_editor_unimportant['layers'] else None

    label_texts = model_labels.get(model_name, [])

    if important_drawing is not None:
        important_drawing = Image.fromarray(important_drawing).convert("RGB")
    if unimportant_drawing is not None:
        unimportant_drawing = Image.fromarray(unimportant_drawing).convert("RGB")

    image_tensor = torch.tensor(original_image, dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    original_image = Image.fromarray(original_image).convert("RGB")
    if important_drawing is not None:
        important_drawing = process_user_drawing(important_drawing)
    if unimportant_drawing is not None:
        unimportant_drawing = process_user_drawing(unimportant_drawing)

    combined_mask = np.full(important_drawing.shape, -1)
    if important_drawing is not None:
        combined_mask[important_drawing > 0] = 1
    if unimportant_drawing is not None:
        combined_mask[unimportant_drawing > 0] = 0

    prompt_refiner = PromptRefinement(num_iterations=100)
    refined_mask = prompt_refiner.refine_prompt(image_tensor, combined_mask, target_class=1)

    # plt.imshow(refined_mask, cmap='gray')
    # plt.title('Refined')
    # plt.savefig("refined_drawing.png")
    # plt.close()
    # processed_drawing = Image.open("refined_drawing.png")
    # processed_drawing.show()

    combined_mask_3d = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.multiply(original_image, combined_mask_3d)
    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))

    # masked_image[masked_image > 0] = 255
    # masked_image_array = np.abs(masked_image)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(masked_image_array)
    # plt.title("Masked Image")
    # plt.axis('off')
    # plt.savefig("processed_drawing.png")
    # plt.close()
    # processed_drawing = Image.open("processed_drawing.png")
    # processed_drawing.show()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    original_image_tensor = transform(original_image).unsqueeze(0)
    masked_image_tensor = transform(masked_image_pil).unsqueeze(0)

    model = load_model(model_name)
    with torch.no_grad():
        outputs_original = model(original_image_tensor)
        outputs_masked = model(masked_image_tensor)

    predicted_label_original = outputs_original.argmax(dim=1).item()
    predicted_label_masked = outputs_masked.argmax(dim=1).item()

    label_text_original = label_texts[predicted_label_original] if label_texts else f"Label {predicted_label_original}"
    label_text_masked = label_texts[predicted_label_masked] if label_texts else f"Label {predicted_label_masked}"

    return f"Original Image Prediction: {label_text_original}\nPrompted Image Prediction: {label_text_masked}"

available_models = list_models(models_folder)

with gr.Blocks() as demo:
    uploaded_image = gr.State()
    with gr.Tabs():
        with gr.Tab("Step 1: Choose Model"):
            with gr.Column():
                model_input = gr.Radio(choices=list(available_models), label="Choose Model")
                labels_output = gr.Textbox(label="Predefined Labels", interactive=False)
                next_button1 = gr.Button("Next: Upload Image")

                model_input.change(
                    update_labels,
                    inputs=[model_input],
                    outputs=[labels_output]
                )

        with gr.Tab("Step 2: Upload Image"):
            with gr.Column():
                image_input = gr.Image(label="Upload Image", image_mode='RGB')
                next_button2 = gr.Button("Next: Highlight Important Areas")

        with gr.Tab("Step 3: Highlight Important Areas"):
            with gr.Column():
                image_editor_important = gr.ImageEditor(label="Highlight Important Areas", image_mode='RGB')
                next_button3 = gr.Button("Next: Highlight Unimportant Areas")

        with gr.Tab("Step 4: Highlight Unimportant Areas"):
            with gr.Column():
                image_editor_unimportant = gr.ImageEditor(label="Highlight Unimportant Areas", image_mode='RGB')
                next_button4 = gr.Button("Next: Classify Image")

        with gr.Tab("Step 5: Classification Result"):
            with gr.Column():
                output_text = gr.Textbox(label="Predicted Labels")
                # classify_button = gr.Button("Classify Image")

        next_button1.click(None, [], [], js="() => {document.querySelectorAll('button')[1].click()}")

        image_input.upload(
            update_image_editor,
            inputs=image_input,
            outputs=[image_editor_important, image_editor_unimportant]
        )

        next_button2.click(
            None,
            inputs=[],
            outputs=[],
            js="() => {document.querySelectorAll('button')[2].click()}"
        )

        next_button3.click(
            None,
            inputs=[],
            outputs=[],
            js="() => {document.querySelectorAll('button')[3].click()}"
        )

        next_button4.click(
            classify_image,
            inputs=[image_editor_important, image_editor_unimportant, model_input],
            outputs=[output_text],
            js="() => {document.querySelectorAll('button')[4].click()}"
        )

        # classify_button.click(
        #     classify_image,
        #     inputs=[image_editor_important, image_editor_unimportant, model_input],
        #     outputs=[output_text]
        # )

demo.launch()