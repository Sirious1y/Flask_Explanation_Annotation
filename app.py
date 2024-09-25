import gradio as gr
import numpy as np
from functions import list_models, update_labels, update_image_editor, refresh_image_editor, classify_image
from utils import models_folder

is_front = True
available_models = list_models(models_folder)

with gr.Blocks(title="Visual Prompted Prediction", css="./styles/app.scss") as demo:
    uploaded_image = gr.State()
    with gr.Tabs():
        with gr.Tab("1.\tSelect Model", elem_id="step-1"):
            with gr.Column():
                gr.Markdown("## Select Model")
                gr.Markdown("Select a model from the list. The predefined labels for the chosen model will be displayed.")
                model_input = gr.Dropdown(choices=list(available_models), label="", value=available_models[0])
                gr.Markdown("## Predefined Labels")
                labels_output = gr.Textbox(label="", interactive=False)

                model_input.change(
                    update_labels,
                    inputs=[model_input],
                    outputs=[labels_output]
                )
                
            next_button1 = gr.Button("Next")

        with gr.Tab("2.\tUpload Image", elem_id="step-2"):
            with gr.Column():
                gr.Markdown("## Upload Image")
                gr.Markdown("Upload the image you want to analyze. This image will be used in the following steps for highlighting important and irrelevant areas.")
                image_input = gr.Image(label="Upload Image", image_mode='RGB')
                
            next_button2 = gr.Button("Next")

        with gr.Tab("3.\tAnnotate Image", elem_id="step-3"):
            with gr.Column():
                gr.Markdown("## Highlight Important Areas")
                gr.Markdown("Highlight the important areas in the uploaded image by drawing on it. These areas will be considered crucial for the decision making process.")
                image_editor_important = gr.ImageEditor(label="Highlight Important Areas", image_mode='RGB')
                # next_button3 = gr.Button("Next")
                
            with gr.Column():
                gr.Markdown("## Highlight Unimportant Areas")
                gr.Markdown("Highlight the irrelevant areas in the uploaded image by drawing on it. These areas will be ignored during the decision making process.")
                image_editor_unimportant = gr.ImageEditor(label="Highlight Unimportant Areas", image_mode='RGB')
            
            next_button3 = gr.Button("Next")

        # with gr.Tab("Step 4: Highlight Unimportant Areas"):
        #     with gr.Column():
        #         gr.Markdown("## Step 4: Highlight Unimportant Areas")
        #         gr.Markdown("Highlight the irrelevant areas in the uploaded image by drawing on it. These areas will be ignored during the decision making process.")
        #         image_editor_unimportant = gr.ImageEditor(label="Highlight Unimportant Areas", image_mode='RGB')
        #         next_button4 = gr.Button("Next: Classify Image")

        with gr.Tab("4.\tClassification Result", elem_id="step-4"):
            with gr.Column():
                gr.Markdown("## Classification Result")
                gr.Markdown("View the classification results based on your inputs and highlighted areas.")
                output_text = gr.Textbox(label="Predicted Labels")
                masked_image_display = gr.Image(label="Masked Image", image_mode='RGB', height=256, width=256)
                classify_button = gr.Button("Classify Image")

        next_button1.click(None, [], [], js="() => {document.querySelectorAll('button')[1].click()}")

        image_input.upload(
            update_image_editor,
            inputs=image_input,
            outputs=[image_editor_important, image_editor_unimportant]
        )

        image_editor_important.clear(
            refresh_image_editor,
            inputs=image_input,
            outputs=[image_editor_important]
        )

        image_editor_unimportant.clear(
            refresh_image_editor,
            inputs=image_input,
            outputs=[image_editor_unimportant]
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

        classify_button.click(
            classify_image,
            inputs=[image_editor_important, image_editor_unimportant, model_input],
            outputs=[output_text, masked_image_display]
        )

demo.launch()
