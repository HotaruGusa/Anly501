import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import sklearn.svm

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

# linear kernel
SVM_Model=LinearSVC(C=10)
SVM_Model.fit(TrainDF, train_label)
print("SVM prediction:\n", SVM_Model.predict(TestDF))
print("Actual:")
print(test_label)

SVM_matrix = confusion_matrix(test_label, SVM_Model.predict(TestDF))
print("\nThe confusion matrix for basic linear SVC is:")
print(SVM_matrix)
print("\n\n")
sns.heatmap(SVM_matrix,annot=True)
plt.show()

# RBF kernel
SVM_Model2=sklearn.svm.SVC(C=10, kernel='rbf',
                           verbose=True, gamma="auto")
SVM_Model2.fit(TrainDF, train_label)

print("SVM prediction:\n", SVM_Model2.predict(TestDF))
print("Actual:")
print(test_label)

SVM_matrix = confusion_matrix(test_label, SVM_Model2.predict(TestDF))
print("\nThe confusion matrix for rbf SVM is:")
print(SVM_matrix)
print("\n\n")
sns.heatmap(SVM_matrix,annot=True)
plt.show()

# Poly kernel
SVM_Model3=sklearn.svm.SVC(C=20, kernel='poly',degree=2,
                           gamma="auto", verbose=True)

print(SVM_Model3)
SVM_Model3.fit(TrainDF, train_label)

print("SVM prediction:\n", SVM_Model3.predict(TestDF))
print("Actual:")
print(test_label)

SVM_matrix = confusion_matrix(test_label, SVM_Model3.predict(TestDF))
print("\nThe confusion matrix for SVM poly d=2  is:")
print(SVM_matrix)
print("\n\n")
sns.heatmap(SVM_matrix,annot=True)
plt.show()
