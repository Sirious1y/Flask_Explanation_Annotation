"""
Initialization Code
"""
import json
import numpy as np
import pandas as pd
from os.path import exists
from datetime import datetime
from pytz import timezone


#*##########

folder_name = 'img_gender'
chunk_id = 'chunk_1'
task_type = 'factual'  # counterfactual or factual
# task_type = 'counterfactual'  # counterfactual or factual

#*##########


tz = timezone('EST')

idx_list = pd.DataFrame(["demo_img_1.jpg", "demo_img_2.jpg", "demo_img_3.jpg", "demo_img_4.jpg", "demo_img_5.jpg"])
df = idx_list.reset_index(drop=True)

def add_col(df):
    df.columns = ['img_idx']
    
    df['attention'] = np.nan
    df['img_check'] = np.nan
    df['matrix_resize'] = np.nan

    # for i in range(1, 5):  # 4 selections
    #     df['a_%s' % i] = np.nan
    
    # for i in range(1, 5):
    #     df['b_%s' % i] = np.nan
        
    # for i in range(1, 5):
    #     df['c_%s' % i] = np.nan

    return df

def make_current(df):
    df_current = pd.DataFrame()
    df_current.loc[0, 'current_order'] = 0
    df_current.loc[0, 'current_idx'] = df['img_idx'][0]
    return df_current

df = add_col(df)
df_current = make_current(df)
df_rnd = df

## Save files for UI
csvname_curr = f'current_{folder_name}_{chunk_id}_{task_type}.csv'
csvname_res = f'results_{folder_name}_{chunk_id}_{task_type}.csv'

result_path = f'output/{csvname_res}'

df_current.to_csv(f"output/{csvname_curr}")
df_rnd.to_csv(f"output/{csvname_res}")

## Save initial backup files for starting over
df_current.to_csv(f"output/init_files/{csvname_curr}")
df_rnd.to_csv(f"output/init_files/{csvname_res}")



