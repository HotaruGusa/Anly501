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
data = pd.read_csv('cleanCSV.csv')
print(type(data))
print(data)

# wordcloud for this csv data
df_text = pd.DataFrame(columns = ['text'])
content = []

with open('cleanCSV.csv', "r") as f:
    text = f.read()
    text = re.sub(r'\b\w{1,3}\b', '', text)
    content.append(text)

df_text["text"] = content
text = df_text['text'].str.cat(sep=' ')
text = text.lower()
text = ' '.join([word for word in text.split()])
wd = WordCloud(collocations=False, max_font_size=50, max_words=100, background_color="white").generate(text)
plt.imshow(wd, interpolation="bilinear")
plt.axis("off")

# Use k-means clustering on the data
k = [2,3,4]
for i in k:
    print("k value is", i , "now")
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    labels = kmeans.labels_
    print(labels)
    centroids = kmeans.cluster_centers_
    print(centroids)
    prediction = kmeans.predict(data)
    print(prediction)

# visualization
data_normalized=(data - data.mean()) / data.std()
print(data_normalized)
NumCols=data_normalized.shape[1]

My_pca = PCA(n_components=2)
data_normalized=np.transpose(data_normalized)
My_pca.fit(data_normalized)

print(My_pca)
print(My_pca.components_.T)
KnownLabels=data.LABEL

# Reformat and view results
Comps = pd.DataFrame(My_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(2)],
                        index=data_normalized.columns
                        )

plt.figure(figsize=(12,12))
plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=100, color="green")

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot Clusters PC 1 and 2",fontsize=15)
for i, label in enumerate(KnownLabels):
    plt.annotate(label, (Comps.iloc[i,0], Comps.iloc[i,1]))


# DBSCAN
MyDBSCAN = DBSCAN(eps=6, min_samples=2)
MyDBSCAN.fit_predict(data)
print(MyDBSCAN.labels_)

# Hierarchical
MyHC = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
FIT=MyHC.fit(data)
HC_labels = MyHC.labels_
print(HC_labels)

plt.figure(figsize =(12, 12))
plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(data, method ='ward')))

EDist=euclidean_distances(data)
print(EDist)

# looking for best k
Sih=[]
Cal=[]
k_range=range(2,8)

for k in k_range:
    k_means_n = KMeans(n_clusters=k)
    model = k_means_n.fit(data)
    Pred = k_means_n.predict(data)
    labels_n = k_means_n.labels_
    R1=metrics.silhouette_score(data, labels_n, metric = 'euclidean')
    R2=metrics.calinski_harabasz_score(data, labels_n)
    Sih.append(R1)
    Cal.append(R2)

print(Sih) ## higher is better
print(Cal) ## higher is better

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(k_range,Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")
ax2.plot(k_range,Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")

plt.show()

