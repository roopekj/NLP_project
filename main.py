import json

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

dataraw = []
labels = []

MAX_DATA = 2000

with open('news.json', 'r') as f:
    lines = f.readlines()
    for line in lines[:MAX_DATA]:
        dataj = json.loads(line)
        headline = lemmatizer.lemmatize(dataj['headline'].lower())
        headline = ' '.join([word for word in headline.split() if word not in stop_words])
        dataraw.append(headline)
        labels.append(dataj['authors'])

dataraw = np.array(dataraw)

vocab = np.array(list(set([word for sentence in dataraw for word in sentence.split()])))

data = np.zeros([dataraw.shape[0], vocab.shape[0]])
for i, sentence in enumerate(dataraw):
    for word in sentence.split():
        data[i, np.where(vocab == word)[0][0]] += 1

distances = np.zeros([data.shape[0], data.shape[0]], dtype=float)

for i, data1 in enumerate(data):
    for j, data2 in enumerate(data[i:]):
        sim = np.dot(data1, data2)/(np.linalg.norm(data1)*np.linalg.norm(data2))

        distances[i, j+i] = 1 - sim
        distances[j+i, i] = 1 - sim

tsne = TSNE(n_components=2, n_iter=5000, n_iter_without_progress=200, perplexity=35)
tsne_result = tsne.fit_transform(distances)
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': labels})

fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)

lim = (tsne_result.min() - 5, tsne_result.max() + 5)
ax.set_xlim(lim)
ax.set_ylim(lim)

ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()