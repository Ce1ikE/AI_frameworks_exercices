import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy

from collections import Counter

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

class SpacyTokenizer:
    def __init__(self, model="en_core_web_sm"):
        self.model = model
        self.nlp = spacy.load(model, disable=["ner", "parser"])

    def __call__(self, text):
        return [
            t.text.lower() 
            for t in self.nlp(text) 
            if not t.is_space and not t.is_punct and not t.is_stop and not t.like_num
        ]
    
class Vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = ["<pad>", "<unk>"]

    def build(self, token_lists):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def encode(self, tokens):
        return [self.word2idx.get(t, 1) for t in tokens]

    def decode(self, ids):
        return [self.idx2word[i] for i in ids]

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,vocab: Vocabulary):
        super().__init__()
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab.word2idx["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.2, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x,lengths):
        embedded = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        out, (h, c) = self.lstm(packed)
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
    
