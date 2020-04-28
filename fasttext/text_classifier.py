#########################################################################################################
# Adapted from: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
#########################################################################################################

import numpy as np
import random

from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
import time



import torch.nn as nn
import torch.nn.functional as F

from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer


from torchtext.datasets import TextClassificationDataset
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator



#####################################################################################################################
# Auxilary functions
#####################################################################################################################

tokenizer = get_tokenizer("basic_english")


def token_iterator(texts, ngrams):
    for text in texts:
        tokens = tokenizer(text)
        yield ngrams_iterator(tokens, ngrams)


def construct_vocab(texts, ngrams):
    vocab = build_vocab_from_iterator(token_iterator(texts, ngrams))
    return vocab


def text_to_tensor(text, vocab, ngrams):
    tokens = ngrams_iterator(tokenizer(text), ngrams=ngrams)
    token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in tokens]))
    tokens = torch.tensor(token_ids)
    return tokens


def make_torchdataset(vocab, texts, labels, ngrams):
    tokens = [text_to_tensor(text, vocab, ngrams) for text in tqdm(texts)]
    pairs = list(zip(labels, tokens))
    return TextClassificationDataset(vocab, pairs, set(labels))

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

#####################################################################################################################
# Model
#####################################################################################################################


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

#####################################################################################################################
# TextClassifier
#####################################################################################################################


class TextClassifier(object):

    def __init__(self, texts, labels, embed_dim, ngrams=3, num_epochs=5, seed=0):

        # set seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


        self.texts = texts
        self.labels = labels
        self.embed_dim = embed_dim
        self.ngrams = ngrams

        # construct vocab
        print('Constructing vocabulary...')
        self.vocab = construct_vocab(texts, ngrams)
        self.vocab_size = len(self.vocab)

        # prepare dataset
        print('Preparing dataset...')
        self.train_dataset = make_torchdataset(self.vocab, texts, labels, ngrams)
        self.num_classes = len(self.train_dataset.get_labels())

        # prepare device ref and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TextClassificationModel(self.vocab_size, self.embed_dim, self.num_classes).to(self.device)

        # loss function & optimization
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)
        self.batch_size = 16

        self.tokenizer = get_tokenizer("basic_english")
        self.ngrams = ngrams



        if num_epochs > 0:
            print('Training model...')
            self.train(self.train_dataset, num_epochs)


    def train_step(self, sub_train_):

        # Train the model
        train_loss = 0
        train_acc = 0
        data = DataLoader(sub_train_, batch_size=self.batch_size, shuffle=True, collate_fn=generate_batch)
        for i, (text, offsets, cls) in enumerate(data):
            self.optimizer.zero_grad()
            text, offsets, cls = text.to(self.device), offsets.to(self.device), cls.to(self.device)
            output = self.model(text, offsets)
            loss = self.criterion(output, cls)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_acc += (output.argmax(1) == cls).sum().item()

        # Adjust the learning rate
        self.scheduler.step()

        return train_loss / len(sub_train_),  train_acc / len(sub_train_)

    def compute_loss(self, data_):
        loss = 0
        acc = 0
        data = DataLoader(data_, batch_size=self.batch_size, collate_fn=generate_batch)
        for text, offsets, cls in data:
            text, offsets, cls = text.to(self.device), offsets.to(self.device), cls.to(self.device)
            with torch.no_grad():
                output = self.model(text, offsets)
                loss = self.criterion(output, cls)
                loss += loss.item()
                acc += (output.argmax(1) == cls).sum().item()

        return loss / len(data_), acc / len(data_)

    def train(self, train_dataset, n_epochs=5):

        min_valid_loss = float('inf')

        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc  = self.train_step(train_dataset)

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print('')

    def predict(self, text_, return_prob=False):

        with torch.no_grad():
            text = text_to_tensor(text_, self.vocab, self.ngrams)
            output = self.model(text, torch.tensor([0]))

            if return_prob:
                return F.softmax(output, 1).detach().numpy()
            else:
                return output.argmax(1).item()

    def get_text_embedding(self, text_):
        with torch.no_grad():
            text = text_to_tensor(text_, self.vocab, self.ngrams)
            return self.model.embedding(text, offsets=torch.LongTensor([0])).detach().numpy()

    def word_in_vocab(self, word):
        return word in self.vocab.stoi



