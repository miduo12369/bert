import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# 设置随机数种子
def seed_everything(seed=4222):
    torch.manual_seed(4222)
    torch.cuda.manual_seed(4222)
    torch.cuda.manual_seed_all(4222)
    torch.backends.cudnn.deterministic = True

seed_everything()

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, model_output, y):
        y_t = y[:, 0]
        y_t_1 = y[:, 1]
        a = model_output[:, 0]
        B = model_output[:, 1]
        return torch.mean((y_t_1 - (a + B * y_t)) ** 2)

# 提取头400个字符和尾200个字符
def extract_head_tail(text):
    head = text[:400]
    tail = text[-200:]
    return head + tail

tokenizer = AutoTokenizer.from_pretrained('./pretrained_models/')

class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=512):
        texts = [extract_head_tail(t) for t in texts]
        self.encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['targets'] = torch.tensor(self.targets[idx])
        return item


class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.model = AutoModel.from_pretrained('./pretrained_models/')
        # #print(self.model)
        # #exit()
        # num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print(f"Number of trainable parameters: {num_params}")
        # exit()
        self.fc1 = nn.Linear(self.model.config.hidden_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        #with torch.no_grad():
        model_output = self.model(input_ids, attention_mask=attention_mask)
        hidden_state = model_output.pooler_output
        x = self.relu(self.fc1(hidden_state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, criterion, optimizer, data_loader, val_loader, epochs=3, device='cpu'):
    model.train()
    model.to(device)
    train_loss=[]
    eval_loss=[]
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader):
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch['targets'].float().to(device))
            loss.backward()
            optimizer.step()
            # print ("Training loss: {}".format(loss), flush=True)
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        
        # Validation
        # print ("Validating...", flush=True)
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch['input_ids'] = batch['input_ids'].to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                outputs = model(batch['input_ids'], batch['attention_mask'])
                val_loss = criterion(outputs, batch['targets'].float().to(device))
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        train_loss.append(avg_loss)
        eval_loss.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    return train_loss, eval_loss

#train model
# Read the data from a Parquet file
train_df = pd.read_parquet('./data/problem1_train.parquet')
valid_df = pd.read_parquet('./data/problem1_valid.parquet')

# Split the DataFrame into texts and targets
train_texts = train_df['Text'].tolist()
train_y = train_df[['Yt', 'Yt+1']].values.tolist()
val_texts = valid_df['Text'].tolist()
val_y = valid_df[['Yt', 'Yt+1']].values.tolist()


# Create datasets and data loaders
train_dataset = TextDataset(train_texts, train_y, tokenizer)
val_dataset = TextDataset(val_texts, val_y, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Instantiate the model, loss function, and optimizer
model = TextClassifier()
criterion = L2Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Train the model
#train(model, criterion, optimizer, train_loader, val_loader, epochs=5, device=device)
train_model_result=train(model, criterion, optimizer, train_loader, val_loader, epochs=5, device=device)

#plot
#print(train_model_result[0])
plt.subplots(figsize=(10,7))
plt.plot(np.arange(5),train_model_result[0],label='train_loss')
plt.plot(np.arange(5),train_model_result[1],label='test_loss')

plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('bert.png')
plt.show()