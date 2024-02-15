import cv2
import pandas as pd
import numpy as np


df = pd.read_csv('./output/done/results_img_gender_chunk_1_factual_20240215_164304_29.csv', index_col=0)

attention = df['attention'][0]

string_list = attention.strip('[]')

# Split the string into individual sublist strings
sublists = string_list.split('], ')

# Initialize an empty list to store the result
list_result = []

# Iterate over each sublist string
for sublist_str in sublists:
    # Remove any remaining brackets
    sublist_str = sublist_str.strip('[]')
    # Split the sublist string into individual elements
    sublist_elements = sublist_str.split(', ')
    # Convert elements to integers if necessary
    sublist = [int(item) for item in sublist_elements]
    # Append the sublist to the result list
    list_result.append(sublist)

attention = np.array(list_result)
print(attention.shape)


# Scale the array to the range of 0-255
grayscale_array = attention * 255

# Convert to uint8 datatype
grayscale_array = grayscale_array.astype(np.uint8)

# Set a threshold value (adjust as needed)
threshold_value = 128

# Apply thresholding
_, black_white_image = cv2.threshold(grayscale_array, threshold_value, 255, cv2.THRESH_BINARY)

# Display or save the image
cv2.imshow('Black and White Image', black_white_image)
cv2.waitKey(0)
cv2.destroyAllWindows()