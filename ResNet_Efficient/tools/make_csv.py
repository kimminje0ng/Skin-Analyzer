import csv
import json
import os
import pandas as pd

path = "./"    # cmd 경로 기준
tmp_file_list = os.listdir(path)
# print(tmp_file_list)

input_file_list = [file for file in tmp_file_list if file.endswith(".jpg")]
# print(input_file_list)


flag = False
id_list = []

for input_file_name in input_file_list:
    id_list.append(input_file_name.rstrip(".jpg"))

    input_file_name = "train_imgs/" + input_file_name
    #print(input_file_name)

df = pd.DataFrame(id_list, columns = ['uid'])
df['img_path'] = input_file_name
df['disease'] = 'level0'
df['disease_code'] = '0'
print(df)
df.to_csv("alopecia_0.csv", index = False)