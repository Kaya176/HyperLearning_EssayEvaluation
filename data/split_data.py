import json
import zipfile
import os


PATH = './1.Training/라벨링데이터/TL_'
types = ['글짓기.zip','대안제시.zip','설명글.zip','주장.zip','찬성반대.zip']
target_student_group = ['중등','고등']
target_student_grade = '2'
filtered_data = []
counter = 0
for folder_name in types:
    print(folder_name)
    with zipfile.ZipFile(PATH+folder_name) as zf:
    #zip 폴더 내 파일 명 리스트 불러오기
        zipfile_namelist = zf.namelist()
        for file in zipfile_namelist:
            json_file = zf.open(file)
            json_file = json.load(json_file)
            student_info = json_file['student']
            if student_info['student_grade'] in ['중등_2학년','고등_2학년']:
                filtered_data.append(json_file)

#save filtered data
with open('filtered_datas.json','w') as fd:
    json.dump(filtered_data,fd)