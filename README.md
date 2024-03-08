# Flask UI for CNN Explanation Annotation
#### Setup Steps:

1. Open Anaconda Prompt (base env.) and go to the project directory: 
```cmd
$ cd <PATH/TO/PROJECT>
```

2. Create a Python virtual environment called "venv_flask_cnn": 
```cmd
$ python -m venv venv_flask_cnn
```

3. Activate the environment:
```cmd
$ venv_flask_cnn\Scripts\activate.bat
```

4. Install required packages by PIP:
```cmd
$ pip install -r requirements.txt
```

5. Initialize back end files by Python (only run this once before each annotation task, e.g., annotating chunk #1 for factual areas):
- Image file IDs, chunk number, task type ("counterfactual" or "factual") variables can be edited in the `init_code.py` file.
```cmd
$ python init_code.py
```

6. Save your CNN model in './model/' folder

7. Start the Flask app in the terminal:
```cmd
$ python app.py
```
- Then go to `http://127.0.0.1:5000/` in the browser.

8. Specify the file name of your model in the text input
- Currently only pytorch's `.pt` and `.pth` model file are supported.
- If you leave the input blank, the application would load a default pretrained ResNet 18 model.

9. Start labeling:
- Simply circle the areas on the image that you think is important for the decision making. 

10. To exit the app and environment:
```cmd
$ [Press CTRL+C to quit]
$ deactivate
```
