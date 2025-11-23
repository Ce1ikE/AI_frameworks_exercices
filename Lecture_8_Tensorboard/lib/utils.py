import unicodedata
import string
import torch
import time
import math
import random
import torch
import spacy

from collections import Counter
from torch.utils.data import Dataset

from .model import device

def timeSince(since: float):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class NLPUtils:
    all_letters = string.ascii_letters + " .,;'"
    # creates a lookup table
    letter_to_index = {c:i for i,c in enumerate(all_letters)}

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    @classmethod
    def unicodeToAscii(cls,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in cls.all_letters
        )

    @classmethod
    def readLines(cls,filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [cls.unicodeToAscii(line) for line in lines]

    # Find letter index from all_letters, e.g. "a" = 0
    @classmethod
    def letterToIndex(cls,letter):
        return cls.all_letters.find(letter)

    @classmethod
    def categoryFromOutput(cls,output, all_categories):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return all_categories[category_i], category_i
    
    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    @classmethod
    def lineToTensor(cls,line: str):
        # line: string of characters
        idxs = torch.tensor([cls.letter_to_index[c] for c in line], device=device)
        # one-hot: (seq_len, 1, n_letters) 
        line_tensor = torch.zeros(idxs.size(0), 1, len(cls.all_letters), device=device)
        line_tensor[torch.arange(idxs.size(0)), 0, idxs] = 1
        return line_tensor
    

    @classmethod
    def randomSample(cls,to_index_map: dict, all_categories: list, category_lines: dict):
        category = random.choice(all_categories) 
        line = random.choice(category_lines[category])

        # TODO: if batching the shape must be : (1, n_categories) => n_categories == batch size
        category_tensor = torch.tensor([to_index_map[category]], device=device)
        # TODO: if batching the shape must be : (seq_len, n_categories, n_letters) => n_categories == batch size
        line_tensor = cls.lineToTensor(line)

        return category, line, category_tensor, line_tensor

class SpacyTokenizer:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model, disable=["ner", "parser", "tagger"])

    def __call__(self, text):
        return [t.text.lower() for t in self.nlp(text) if not t.is_space]
    

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


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab

        self.encoded = [vocab.encode(tokenizer(t)) for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def pad_collate(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)

    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    return padded, torch.stack(labels), torch.tensor(lengths)