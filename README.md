# HyperLearning_EssayEvaluation

본 저장소는 에세이 자동 평가 시스템 개발 프로젝트의 소스코드를 저장하는 저장소 입니다.

에세이 자동 평가 시스템은 RoBERTa-base 모델을 기반으로 만들었으며, 11개의 소분류를 분류하기 위해 3개의 대분류 분류기를 각각 구현하여 모델을 구성하였습니다. 모델의 구현은 Pytorch 와 HuggingFace를 사용하여 구현하였으며, 평가 지표로는 Micro F1 score을 사용하여 모델을 평가하였습니다.

## 파일 구성
파일 구성은 다음과 같습니다.

📦HyperLearning_EssayEvaluation  
 ┣ 📂data  
 ┃ ┣ 📜data_analysis.ipynb  
 ┃ ┣ 📜make_dataset.py  
 ┃ ┗ 📜split_data.py  
 ┣ 📂docs  
 ┃ ┗ 📜결과 보고서.pdf  
 ┣ 📂src  
 ┃ ┗ 📜model.py  
 ┣ 📜README.md  
 ┗ 📜requirements.txt  

 1. data  
1.1. split_data.py : raw 데이터에서 중고등학교 2학년 생의 데이터만 추출합니다.  
1.2. make_dataset.py : 학습에 필요한 dataframe을 만드는 파일입니다.  
1.3. data_analysis.ipynb : 데이터 분석 과정이 담긴 ipynb 파일입니다.
 2. docs  
 2.1. 결과 보고서.pdf
 3. src  
 3.1. model.py : 모델학습 구현이 담긴 파일입니다.

 ## Get Started
### 0. requirements.txt
```
pip install -r requirements.txt
```
### 1. 학습 데이터 만들기
 ```
 python -m split_data.py
 python -m make_dataset.py
 ```

 ### 2. 모델학습 및 모델 저장
 ```
 python -m model.py
 ```
