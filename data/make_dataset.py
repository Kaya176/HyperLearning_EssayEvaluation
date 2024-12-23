import json
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter

#utils
#column name: paragrah
#여러개의 paragraph를 하나의 문장으로 연결
def merge_txt(paragraph):
    txt = ''
    for p in paragraph:
        txt += p['paragraph_txt']
    return txt

#hard voting
def hard_voting(arr):
    result = []
    for i in range(len(arr[0,:])):
        common = Counter(arr[:,i]).most_common(n = 3)
        if len(common) == 3:
            result.append(sorted(arr[:,i])[1]) #median
        else:
            result.append(common[0][0])
    return result

#가중합
def weighted_sum(scores, weights):
    return sum([s * w for s, w in zip(scores, weights)])

#3명의 평가자별 전체 표현 점수 목록 가져오고 하나의 값으로 만든 뒤, 새로운 최종 점수을 계산함
def calculate_score(score,rubric):
    score = score['essay_scoreT_detail']
    s_exp = hard_voting(np.array(score['essay_scoreT_exp'])) #표현
    s_org = hard_voting(np.array(score['essay_scoreT_org'])) #구조
    s_cont = hard_voting(np.array(score['essay_scoreT_cont'])) #내용

    #평가 가중치
    e_exp = rubric['expression_weight']
    e_org = rubric['organization_weight']
    e_cont = rubric['content_weight']

    weight = {'exp' : list(map(int,([e_exp['exp_grammar'],e_exp['exp_vocab'],e_exp['exp_style']]))),
              'org' : list(map(int,[e_org['org_essay'],e_org['org_paragraph'],e_org['org_coherence'],e_org['org_quantity']])),
              'cont': list(map(int,[e_cont['con_clearance'],e_cont['con_description'],e_cont['con_prompt'],e_cont['con_novelty']]))
             }

    score_exp = weighted_sum(s_exp,weight['exp'])
    score_org = weighted_sum(s_org,weight['org'])
    score_cont = weighted_sum(s_cont,weight['cont'])

    final_score = (score_exp/sum(weight['exp']))*e_exp['exp'] + (score_org/sum(weight['org']))*e_org['org'] + (score_cont/sum(weight['cont']))*e_cont['con']
    
    return [s_exp,s_org,s_cont],final_score

with open('filtered_datas.json') as json_file:
    total_data = json.load(json_file)
json_df = pd.DataFrame(total_data)
data = pd.DataFrame()

#paragraph 갯수
paragraph_num = []
for data in total_data:
    p_num = len(data['paragraph'])
    paragraph_num.append(p_num)

score_list = []
total_score = []
student_group_list = []
student_reading_list = []
student_educated_list = []
type_list = []
#simple function
group2num = lambda x :1 if x['student']['student_grade_group'] == '중등' else 2 #1 : 중등 / 2 : 고등
edu2num = lambda x : 1 if x['student']['student_educated'] else 0 #1 : True / 0 : False
type2type = lambda x: 1 if x['info']['essay_type'] in ['글짓기','설명글'] else 2 #1 : 수필형 / 2 : 논술형

#make dataframe
for idx in range(len(json_df)):
    line = json_df.iloc[idx]
    #score
    score,t_score = calculate_score(line['score'],line['rubric'])
    score_list.append(score)
    total_score.append(t_score)
    #student
    student_group_list.append(group2num(line))
    student_reading_list.append(line['student']['student_reading'])
    student_educated_list.append(edu2num(line))
    #info
    type_list.append(type2type(line))

score_list = np.array(score_list,dtype = object)
score_exp = score_list[:,0]
score_org = score_list[:,1]
score_cont = score_list[:,2]

#다루기 쉬운 dataframe으로 변환하기
df = pd.DataFrame({'paragraph' : json_df['paragraph'].apply(merge_txt),
                   'paragraph_num' : paragraph_num,
                   'score_exp' : score_exp,
                   'score_org' : score_org,
                   'score_con' : score_cont,
                   'total_score' : total_score,
                   'student_group' : student_group_list,
                   'student_reading' : student_reading_list,
                   'info' : json_df['info'],
                   'type' : type_list})
#파일 확인용
#df.to_csv('filtered_data.csv',encoding = 'utf-8-sig',index = False)
#save
df.to_pickle('filtered_datas.pkl')