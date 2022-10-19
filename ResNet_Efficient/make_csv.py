import csv
import json
import os

path = "./"
tmp_file_list = os.listdir(path)
input_file_list = [file for file in tmp_file_list if file.endswith(".json")]

flag = False
for input_file_name in input_file_list:
    print(input_file_name)
    output_file_name = "test.csv"

    with open(input_file_name, "r", encoding="utf-8", newline="") as input_file, \
            open(output_file_name, "a", encoding="utf-8", newline="") as output_file:
        
        data = []
        for line in input_file:
            datum = json.loads(line)
            
            datum['uid'] = datum.pop('image_id')                # uid
            datum['img_path'] = datum.pop('image_file_name')    # 파일명
            datum['disease'] = 'hairloss'                       # 탈모      -> 수정 필요
            datum['disease_code'] = datum.pop('value_6')        # 질병코드  -> 수정 필요
            del datum['value_1']; del datum['value_2']; del datum['value_3']; del datum['value_4']; del datum['value_5']
            
            data.append(datum)
            
        csvwriter = csv.writer(output_file)

        if flag == False:
            csvwriter.writerow(data[0].keys())
        flag = True

        for line in data:
            csvwriter.writerow(line.values())