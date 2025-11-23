import torch
import torch.nn as nn
import torch.nn.functional as F

cudnn = 'cudnn' if torch.backends.cudnn.is_available() else False 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        rnn_out, hidden = self.rnn(input)
        output = self.h2o(hidden[0])
        return self.softmax(output)


## Homework
# -  Try with a different dataset of line -> category, for example:
#    -  Any word -> language
#    -  First name -> gender
#    -  Character name -> writer
#    -  Page title -> blog or subreddit
# -  Get better results with a bigger and/or better shaped network

#    -  Add more linear layers
#    -  Try the ``nn.LSTM`` and ``nn.GRU`` layers
#    -  Combine multiple of these RNNs as a higher level network

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,vocab):
        super().__init__()
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        e = self.embed(x)
        out, (h, c) = self.lstm(e)
        h_last = h[-1]
        logits = self.fc(h_last)
        return logits
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    