import numpy as np
import re

def load_data(file):
    if file == "news":
        file = "../news_data.npz"
    else:
        file = "../spam_data.npz"
    data = np.load(file, allow_pickle=True)
    train, test = {}, {}
    train['texts'], train['labels'] = data['train_texts'], data['train_labels']
    test['texts'], test['labels'] = data['test_texts'], data['test_labels']

    return train, test

def load_similar_news(filename):
    new_news = {}
    texts = []
    labels = []
    with open(filename,'r', encoding='utf-8') as file:
        for line in file:
            elements = line.split();
            labels.append(int(elements[0]))
            text = (" ".join(elements[1:]))
            texts.append(text)
    new_news['texts'] = np.asarray(texts)
    new_news['labels'] = np.asarray(labels)
    return new_news

