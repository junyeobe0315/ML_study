import numpy as np
import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary as summary_
import torchvision
from tqdm import tqdm

config = {
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 1,
        "num_epoch": 1,
        "learning_rate": 0.001
    }
}

data = pd.read_csv('./spam.csv',encoding='latin1')

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])

data.drop_duplicates(subset=['v2'], inplace=True)

X_data = data['v2']
Y_data = data['v1']

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded = tokenizer.texts_to_sequences(X_train)
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index) + 1

max_len = 189
X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len)

X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len)

class BasicDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(BasicDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

X_train_padded_tensor = torch.from_numpy(X_train_padded).float()
X_train_padded_tensor = torch.tensor(X_train_padded_tensor, dtype=torch.float32)

X_test_padded_tensor = torch.from_numpy(X_test_padded).float()
X_test_padded_tensor = torch.tensor(X_test_padded_tensor, dtype=torch.float32)

Y_train_np = Y_train.values
Y_train_tensor = torch.from_numpy(Y_train_np).float()
Y_train_tensor = torch.tensor(Y_train_tensor, dtype=torch.long)

Y_test_np = Y_test.values
Y_test_tensor = torch.from_numpy(Y_test_np).float()
Y_test_tensor = torch.tensor(Y_test_tensor, dtype=torch.long)

train_dataset = BasicDataset(X_train_padded_tensor, Y_train_tensor)
test_dataset = BasicDataset(X_test_padded_tensor, Y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(189, 64, batch_first=True)
        self.linear = nn.Linear(64, 2)

    def forward(self, x):
        x_out, x_hidden = self.rnn(x)
        result = self.linear(x_hidden)
        return result

model = RNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def run_epoch(dataloader, model, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = 0.001

    return epoch_loss, lr

loss_list = []

for epoch in tqdm(range(config["training"]["num_epoch"])):
    loss_train, lr_train = run_epoch(train_loader, model, is_training=True)
    loss_list.append(loss_train)

total_correct = 0
total_data = 0

for batch_idx, (data, target) in enumerate(test_loader):
    
    inputs = data
    target = target
    out = model(inputs)

    result = torch.argmax(out[0])
    num_correct = torch.eq(result, target).sum().item()
    num_data = len(target)

    total_correct += num_correct
    total_data += num_data

print('accuracy :', total_correct/total_data)
print(model)