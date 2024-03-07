from flask import Flask, render_template, session, request, redirect, url_for, flash
from forms import ReasonForm
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
    return render_template('home.html')

@app.route("/index", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if len(request.form['model_name']) == 0:
            session['model_path'] = None
        else:
            model_path = './model/' + request.form['model_name']
            if os.path.isfile(model_path):
                session['model_path'] = model_path
            else:
                flash('Model does not exist. Please double check the name you entered. ')
                return redirect(url_for('home'))

        form = ReasonForm()
        result_path = f'output/{csvname_res}'
        try:
            if True:
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

                if form.validate_on_submit():
                    svg_path_input = request.form['crop_input']  # PNG-base64 string
                    map_all = pil2bin_matrix(svg_path_input)

                    try:
                        map_all = (cv2.resize(np.float32(map_all), matrix_resize_w_h) > 0) * 1  # (width, height)
                        results.append(json.dumps(map_all.tolist()))  # narray/list to string
                        results.append("good")
                        results.append(str(matrix_resize_w_h))

                    except:
                        results.append(map_all)  # skipped
                        results.append(svg_path_input)  # skipped
                        results.append(str(matrix_resize_w_h))

                    df_results.loc[current_order, :len(results)+1] = [current_idx] + results
                    print(f'map_all: {map_all}')

                    if current_order + 1 < len(idx_list):
                        current_order += 1
                        current_idx = idx_list[current_order].split('.jpg')[0]

                    else:
                        finish_text = f"Thank You! Your verification code is: {verify_code} (only appears once)"
                        form_display = False
                        os.remove(result_path)
                        timestr = datetime.now(tz).strftime("%Y%m%d_%H%M%S_%f")[:-4]
                        result_path = f"output/done/results_{folder_name}_{chunk_id}_{task_type}_{timestr}.csv"
                        pass

                    df_results.to_csv(result_path)

                else:
                    pass

                print('reached 113')
                #### Radio default selections

                form.reason_input_1.data = 'True'
                form.reason_input_2.data = 'True'
                form.reason_input_3.data = 'True'
                form.reason_input_4.data = 'True'
                form.reason_input_5.data = 'True'
                form.reason_input_6.data = 'True'
                form.reason_input_7.data = 'True'
                form.reason_input_8.data = 'True'
                form.reason_input_9.data = '5'
                form.reason_input_10.data = '5'
                form.reason_input_11.data = '5'
                form.reason_input_12.data = '5'


                df_current.loc[0, 'current_order'] = current_order
                df_current.loc[0, 'current_idx'] = idx_list[current_order]

                df_current.to_csv(f'output/{csvname_curr}')

                q1_q2_display = False
                q3_display = False

                if q1_q2_display == False:
                    q1_q2_display = "display: none;"
                else:
                    q1_q2_display = ""

                if q3_display == False:
                    q3_display = "display: none;"
                else:
                    q3_display = ""

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

                return render_template('index.html', title='Main', current_order=str(current_order+1), len_idx = len(idx_list), finish_text=finish_text, img_paths=img_paths, form=form, form_display=form_display, q1_q2_display=q1_q2_display, q3_display=q3_display, draw_display=draw_display, drawing_pad_size=drawing_pad_size, task_title=task_title, task_counter_display=task_counter_display, task_fact_display=task_fact_display)

        except:
            return render_template('blank.html', title='Blank')
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

        img_paths = session.get('img_path', None)
        ml_model = model.ModelClass()
        model_path = session.get('model_path', None)
        ml_model.load_model(model_path)

        image = model.load_image(img_paths[0])
        image_tensor = model.process_image(image)
        image_tensor = model.masking(image_tensor, map_all)
        print(image_tensor.shape)
        image_name = img_paths[0].split('/')[-1]
        print(image_name)
        anno_path = './static/images/annotated/annotated_' + image_name
        model.save_image(image, map_all, anno_path)

        # print(image_tensor.shape)
        result = ml_model.predict(image_tensor)

    return render_template('result.html', title='Result', task_title='test', result=result, img_path=img_paths[0], draw_display='', anno_path=anno_path)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)