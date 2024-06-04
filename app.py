from flask import Flask, render_template, session, request, redirect, url_for, flash
from flask_wtf import FlaskForm
import os
import pandas as pd
import json
from datetime import datetime
from pytz import timezone
from config import SECRET_KEY

## Run initialization code "init_code.py"
# import init_code
from init_code import chunk_id, folder_name, csvname_curr, csvname_res, tz, task_type

## SVG path to matrix
from io import BytesIO
import cv2
from PIL import Image
import base64
import numpy as np
# np.set_printoptions(threshold=np.inf)

import model


#*##########

verify_code = '######'
drawing_pad_size = (400, 400)  # height, width in px
matrix_resize_w_h = (224, 224)  # height, width in px

#*##########


def pil2bin_matrix(img_src):
    def base64_prep(img_src):
        try:
            output = img_src.split(",")[1]
        except:
            output = ""

        return output
    
    if base64_prep(img_src) != "": 
        pil_map = Image.open(BytesIO(base64.b64decode(base64_prep(img_src))))
        ## Convert PIL RGB to binary martix
        overlap_matrix = cv2.cvtColor(np.float32(pil_map), cv2.COLOR_RGB2GRAY)
        return (overlap_matrix > 0) * 1
    else:
        return "skipped"


app = Flask(__name__)

app.config['SECRET_KEY'] = SECRET_KEY
# tz = timezone('EST')


@app.route("/", methods=['GET'])
def home():
    models = os.listdir('./model')
    models = list(filter(lambda x: x.endswith('.pt') or x.endswith('.pth'), models))
    return render_template('home.html', title='Main', models=models)

@app.route("/index", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'model_name' not in request.form or 'model_name_non_visual' not in request.form:
            print(session.get('model_path'))
        elif len(request.form['model_name']) == 0 or len(request.form['model_name_non_visual']) == 0:
            flash('You must select a model to proceed.')
            return redirect(url_for('home'))
        else:
            model_path = './model/' + request.form['model_name']
            non_visual_path = './model/' + request.form['model_name']
            if os.path.isfile(model_path) and os.path.isfile(non_visual_path):
                session['model_path'] = model_path
                session['non_visual_path'] = non_visual_path
            else:
                flash('Model does not exist. Please double check the name you entered. ')
                return redirect(url_for('home'))

        form = FlaskForm()
        print(f'hidden tags: {form.hidden_tag()}')
        result_path = f'output/{csvname_res}'
        # try:
        if True:
            # if 'current_order' in session and 'current_image' in session:
            #     current_order = session['current_order']
            #     current_idx = session['current_image']
            #     df_results = pd.read_csv(result_path, index_col=0)
            #     idx_list = df_results['img_idx']
            # else:
            df_results = pd.read_csv(result_path, index_col=0)
            df_current = pd.read_csv(f'output/{csvname_curr}', index_col=0)
            idx_list = df_results['img_idx']
            current_order = int(df_current['current_order'])
            current_idx = idx_list[current_order].split('.jpg')[0]
            finish_text = ""
            results = []
            form_display = True

            img_paths = []

            img_paths.append(f'static/images/{folder_name}/{current_idx}.jpg')

            session['img_path'] = img_paths

            # print(current_order)
            # print(current_idx)
            # print(idx_list)
            if current_order + 1 < len(idx_list):
                current_order += 1
                current_idx = idx_list[current_order].split('.jpg')[0]
                session['current_order'] = current_order
                session['current_image'] = current_idx
                df = pd.DataFrame({'current_order': [current_order],
                                   'current_idx': [current_idx]})
                df.to_csv(f'output/{csvname_curr}', index=False)


            df_current.loc[0, 'current_order'] = current_order
            df_current.loc[0, 'current_idx'] = idx_list[current_order]

            df_current.to_csv(f'output/{csvname_curr}')


            if form_display == False:
                draw_display = "display: none;"
            else:
                draw_display = ""

            if task_type == 'factual':
                task_counter_display = "display: none;"
                task_fact_display = ""
            else:
                task_counter_display = ""
                task_fact_display = "display: none;"

            task_title = folder_name.split("_")[1] + "_" + chunk_id + "_" + task_type

            return render_template('index.html', title='Main', current_order=str(current_order), len_idx = len(idx_list), finish_text=finish_text, img_paths=img_paths, form=form, form_display=form_display, draw_display=draw_display, drawing_pad_size=drawing_pad_size, task_title=task_title, task_counter_display=task_counter_display, task_fact_display=task_fact_display)

        # except:
        #     return render_template('blank.html', title='Blank')
    else:
        flash('Invalid! Please set the model and class label.')
        return redirect(url_for('home'))


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        svg_path_input = request.form['crop_input']  # PNG-base64 string
        map_all = pil2bin_matrix(svg_path_input)
        results = []
        try:
            map_all = (cv2.resize(np.float32(map_all), matrix_resize_w_h) > 0) * 1  # (width, height)
            results.append(json.dumps(map_all.tolist()))  # narray/list to string
            results.append("good")
            results.append(str(matrix_resize_w_h))

        except:
            results.append(map_all)  # skipped
            results.append(svg_path_input)  # skipped
            results.append(str(matrix_resize_w_h))

        # load models
        visual_model = model.ModelClass()
        non_visual_model = model.ModelClass()
        model_path = session.get('model_path', None)
        non_visual_path = session.get('non_visual_path', None)
        visual_model.load_model(model_path)
        non_visual_model.load_model(non_visual_path)

        # load input image
        img_paths = session.get('img_path', None)
        image = model.load_image(img_paths[0])
        image_tensor = model.process_image(image)
        annotated_image_tensor = model.masking(image_tensor, map_all)
        print(image_tensor.shape)
        print(annotated_image_tensor)
        image_name = img_paths[0].split('/')[-1]
        print(image_name)
        anno_name = image_name.split('.')[0]
        anno_name += '_annotated.jpg'
        print(anno_name)
        anno_path = './static/images/annotated/' + anno_name
        model.save_image(image, map_all, anno_path)

        # print(image_tensor.shape)
        visual_result = visual_model.predict(annotated_image_tensor)
        non_visual_result = non_visual_model.predict(image_tensor)
        print(visual_result)
        print(non_visual_result)
        result = [model.get_result(visual_result), model.get_result(non_visual_result)]

    return render_template('result.html', title='Result', task_title='test', result=result, img_path=img_paths[0], draw_display='', anno_path=anno_path)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, use_reloader=True)
