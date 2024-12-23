import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from sklearn.metrics import f1_score
# 데이터 로드
data_path = "C:/Users/Kaya/JupyterNotebook/hyperlearning/data/filtered_datas.pkl"
data = pd.read_pickle(data_path)
paragraph = data["paragraph"]
labels = np.array(data[["score_exp", "score_org", "score_con"]].apply(lambda row: [item for sublist in row for item in sublist], axis=1).tolist())

# 데이터셋 정의
class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def text_preprocessing(self,text):
        #마침표 제거
        text = re.sub(r"[.]",'',text)
        #문장 구분 표시 제거('#@문장구분#' -> '.')
        text = '.'.join([t.strip() for t in text.split(sep = '#@문장구분#')])
        #줄바꿈 제거
        text = re.sub('\n','',text)
        #특수문자 제거
        text = re.sub(r"[^.\uAC00-\uD7A30-9a-zA-Z\s]",'',text)
        #띄어쓰기 조절
        text = ' '.join([i.strip() for i in text.split()])
        return text
        
    def __getitem__(self, idx):
        text = self.text_preprocessing(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
        }

# 토크나이저 준비
from transformers import AutoModel,AutoTokenizer

pre_trained_model = "roberta-base"
max_len = 128
batch_size = 32
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
dataset = EssayDataset(paragraph, labels, tokenizer, max_len)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class EssayScorer(nn.Module):
    def __init__(self, pretrained_model_name):
        super(EssayScorer, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # 공유 레이어 & Dropout
        self.shared_layer = nn.Linear(self.bert.config.hidden_size, 128)
        self.dropout = nn.Dropout(0.3)
        
        # 각 점수를 예측하는 개별 분류기
        self.exp_classifier = nn.Linear(128, 3)  # score_exp
        self.org_classifier = nn.Linear(128, 4)  # score_org
        self.con_classifier = nn.Linear(128, 4)  # score_con

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        shared_output = self.dropout(torch.relu(self.shared_layer(bert_output.last_hidden_state)))
        shared_output = shared_output[:, 0, :]
        # 각 점수별 분류기 출력
        exp_output = self.exp_classifier(shared_output)
        org_output = self.org_classifier(shared_output)
        con_output = self.con_classifier(shared_output)
        
        return exp_output, org_output, con_output


model = EssayScorer(pretrained_model_name=pre_trained_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 학습 루프
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        exp_output, org_output, con_output = model(input_ids, attention_mask)

        # 손실 계산
        loss_exp = criterion(exp_output, labels[:, :3])
        loss_org = criterion(org_output, labels[:, 3:7])
        loss_con = criterion(con_output, labels[:, 7:])
        loss = loss_exp + loss_org + loss_con

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def calculate_accuracy(predictions, labels):

    # 정확도 계산
    preds_numpy = predictions.round()
    total_f1_list = [f1_score(y_true,y_pred,average = 'micro') for y_true,y_pred in zip(labels,preds_numpy)]
    overall_accuracy = sum(total_f1_list)/len(total_f1_list)
    #카테고리별 micro f1 score 계산
    exp_f1_list = [f1_score(y_true,y_pred,average = 'micro') for y_true,y_pred in zip(labels[:3],preds_numpy[:3])]
    exp_f1_score = sum(exp_f1_list)/len(exp_f1_list)

    org_f1_list = [f1_score(y_true,y_pred,average = 'micro') for y_true,y_pred in zip(labels[3:7],preds_numpy[3:7])]
    org_f1_score = sum(org_f1_list) / len(org_f1_list)

    con_f1_list = [f1_score(y_true,y_pred,average = 'micro') for y_true,y_pred in zip(labels[7:],preds_numpy[7:])]
    con_f1_score = sum(con_f1_list) / len(con_f1_list)
    
    category_f1_score = {"Expression" : exp_f1_score, "Organization":org_f1_score, "contents" : con_f1_score}

    return overall_accuracy,category_f1_score

def evaluate_model_with_accuracy(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            exp_output, org_output, con_output = model(input_ids, attention_mask)
            combined_output = torch.cat([exp_output, org_output, con_output], dim=1)
            loss = criterion(combined_output, labels)
            total_loss += loss.item()

            # 예측값과 실제값 저장
            all_predictions.append(combined_output.cpu().numpy())
            all_labels.append(labels.cpu().numpy())


    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return total_loss / len(dataloader), all_predictions, all_labels

# 학습 및 평가 실행
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, all_predictions, category_accuracies = evaluate_model_with_accuracy(model, val_loader, criterion, device)
    # 정확도 계산
    overall_accuracy,category_score = calculate_accuracy(all_predictions, category_accuracies)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Overall f1 score: {overall_accuracy}")
    print(f"F1 score detail: {category_score}")


# 저장
torch.save(model.state_dict(), "essay_scorer_model.pt")