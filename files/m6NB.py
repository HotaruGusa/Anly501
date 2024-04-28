import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cleanCSV1.csv')
df.drop("marvel", axis=1, inplace=True)
df.drop("universe", axis=1, inplace=True)
print(df)

TrainDF, TestDF = train_test_split(df, test_size=0.3)
train_label = TrainDF['LABEL']
test_label = TestDF['LABEL']
TrainDF = TrainDF.drop(["LABEL"],axis = 1)
TestDF = TestDF.drop(["LABEL"],axis = 1)

MyModelNB= MultinomialNB()
NB=MyModelNB.fit(TrainDF, train_label)
Prediction = MyModelNB.predict(TestDF)
print(np.round(MyModelNB.predict_proba(TestDF),2))

cnf_matrix = confusion_matrix(test_label, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot=True)
plt.show()

MyModelNB= GaussianNB()
NB=MyModelNB.fit(TrainDF, train_label)
Prediction = MyModelNB.predict(TestDF)
print(np.round(MyModelNB.predict_proba(TestDF),2))

cnf_matrix = confusion_matrix(test_label, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
sns.heatmap(cnf_matrix,annot=True)
plt.show()
