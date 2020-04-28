import numpy as np

def load_data(file):
    if file == "news":
        file = "../news_data.npz"
    else:
        file = "../spam_data.npz"
    data = np.load(file)
    train, test = {}, {}
    train['texts'], train['labels'] = data['train_texts'], data['train_labels']
    test['texts'], test['labels'] = data['test_texts'], data['test_labels']

    return train, test