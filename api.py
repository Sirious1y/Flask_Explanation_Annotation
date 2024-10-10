import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from functions import classify_image
from utils import get_labels, list_models, load_model, process_user_drawing
from refinement import PromptRefinement
from config import model_folder

app = Flask(__name__)

model_name_list = list_models(model_folder)
model_list = {}

for model_name in model_name_list: 
    model = load_model(model_name)
    model_list[model_name] = model

@app.route('/predict', methods=['POST'])
def predict(): 
    data = request.json
    original_image = np.array(data['original'], dtype=np.uint8)
    important_drawing = np.array(data['important'], dtype=np.uint8)
    unimportant_drawing = np.array(data['unimportant'], dtype=np.uint8)
    model_name = data['model_name']

    result, masked_image = classify_image(original_image, important_drawing, unimportant_drawing, model_name, model_list[model_name])

    return jsonify({'result': result, 'masked': masked_image.tolist()})

if __name__ == '__main__': 
    app.run(debug=True)

