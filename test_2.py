import numpy as np
import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential
import torch.nn as nn
import torch


data = pd.read_csv('./spam.csv',encoding='latin1')
print('number of sample :', len(data))

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
print(data.info())
data.drop_duplicates(subset=['v2'], inplace=True)
print('number of sample :', len(data))
X_data = data['v2']
Y_data = data['v1']
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0, stratify=Y_data)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded = tokenizer.texts_to_sequences(X_train)
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index) + 1

max_len = 189
X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train_padded, Y_train, epochs=5, batch_size = 64, validation_split=0.2)
print(model.summary())
print(X_train_padded.shape)
print(vocab_size)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(vocab_size, 32, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        out, hidden = self.rnn(x)
        result = self.linear(hidden)
        return result

model = RNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 5
loss_list = []
model.cuda()



X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len)

print('\n test accuracy : % 4f' % (model.evaluate(X_test_padded, Y_test)[1]))


