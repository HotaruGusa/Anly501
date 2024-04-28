import re
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# show the data
data = pd.read_csv('newsapi.csv')
data = data['Headline']
data = data.str.lower()

deleteWord = ["with", "just", "within", "that"]

for sentence in data:
    word = re.split('\s+', sentence)
    with open('news_transaction.csv', 'a') as f:
        for w in word:
            if w not in deleteWord:
                f.write(w)
                f.write(",")
        f.write("\n")
