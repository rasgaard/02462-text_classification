from scipy.linalg import eigh
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import re
import pandas as pd

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

def load_emails(filename):
    emails = {}
    df = pd.read_csv(filename)
    emails['texts'] = np.asarray(df['text'])
    emails['labels'] = np.asarray(df['spam'])
    return emails


def pca(X_train_emb):
    #pca on feature vectors for selected words
    mean_vector = np.mean(X_train_emb, axis=1)
    data = X_train_emb - mean_vector[:,None]

    #compute the covariance matrix
    S = np.cov(data)
    
    #obtain eigenvectors and eigenvalues
    eigenValues, eigenVectors = eigh(S)
    
    #sort according to size of eigenvalues
    sort_idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[sort_idx]
    eigenVectors = eigenVectors[:, sort_idx]
    return eigenValues, eigenVectors, data



def get_train_emb(X_train, embedding, embed_dim=50):
    return np.array([embedding(text) for text in X_train]).T.reshape(embed_dim, -1)


def plot_pca(pca, y_train, PC_range, num_texts=None):
    eigenValues, eigenVectors, data = pca
    X_proj = eigenVectors[:,PC_range[0]:PC_range[1]].T@data
    
    #plot for the selected two principal components
    n_label = len(np.unique(y_train))
    colors = cm.rainbow(np.linspace(0, 1, n_label))
    class_idx = y_train[:num_texts]

    if n_label == 2:
        considered_classes = ['not-spam','spam']
    else:
        considered_classes = ['World','sports','Business', 'Sci/Tec']
    
    cdict = {i: colors[i] for i in range(n_label)}
    label_dict = {i: considered_classes[i] for i in range(n_label)}
    plt.figure(figsize=(10,7))
    for i in range(n_label):
        indices = np.where(class_idx == i)
        plt.scatter(X_proj[0,indices], X_proj[1,indices], color=cdict[i], label=label_dict[i], s=10)

    plt.legend(loc='best')
    plt.xlabel('Principal Component axis 1')
    plt.ylabel('Principal Component axis 2')


def pca_variance_plots(eigenValues, output_total=False):
    embed_dim = eigenValues.shape[0]
    summ = eigenValues.sum()
    cumsum = 0
    total_var_explained = np.zeros(embed_dim)
    relative_var = np.zeros(embed_dim)
    for i in range(embed_dim):
        relative_var[i] = eigenValues[i]/np.size(eigenValues)
        cumsum += eigenValues[i]
        total_var_explained[i]=(cumsum/summ)

    plt.figure(figsize=(12,3))
    plt.subplot(121)
    # this might not be calculated correctly (line 10)
    plt.plot(relative_var)
    plt.xlabel("Principal component")
    plt.ylabel("Proportion of Variance Explained")
    plt.title('variance explained')
    plt.subplot(122)
    plt.plot(total_var_explained)
    plt.xlabel("Principal component")
    plt.ylabel("% of variance explained")
    plt.title('Cumulative variance explained')
    if output_total:
        return total_var_explained