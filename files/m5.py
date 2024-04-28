import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import warnings
warnings.filterwarnings('ignore')

# read csv and part of cleanning
df = pd.read_csv('cleanCSV.csv')
df.drop("marvel", axis=1, inplace=True)
df.drop("universe", axis=1, inplace=True)

List_of_WC = []

topics = ["new", "middle", "old"]

# WordCloud
for mytopic in topics:
    tempdf = df[df['LABEL'] == mytopic]
    tempdf =tempdf.sum(axis=0,numeric_only=True)
    NextVarName=str("wc"+str(mytopic))
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                            min_word_length=4,
                            max_words=200).generate_from_frequencies(tempdf)
    List_of_WC.append(NextVarName)

fig=plt.figure(figsize=(25, 25))
NumTopics=len(topics)

for i in range(NumTopics):
    ax = fig.add_subplot(NumTopics,1,i+1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.savefig("NewClouds.pdf")

# Decision Tree
TrainDF, TestDF = train_test_split(df, test_size=0.25)

TestLabels = TestDF["LABEL"]
TestDF = TestDF.drop(["LABEL"], axis=1)
TrainLabels = TrainDF["LABEL"]
TrainDF = TrainDF.drop(["LABEL"], axis=1)

# first decision tree
MyDT=DecisionTreeClassifier(criterion='entropy',
                            splitter='best',
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=None,
                            random_state=None,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None,
                            class_weight=None)

MyDT.fit(TrainDF, TrainLabels)
feature_names=TrainDF.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                                   feature_names=feature_names,
                                   class_names=topics,
                                   filled=True, rounded=True,
                                   special_characters=True)

graph = graphviz.Source(Tree_Object)
graph.render("Mytree1")

## Confusion Matrix
DT_pred = MyDT.predict(TestDF)
bn_matrix = confusion_matrix(TestLabels, DT_pred)
print("\nThe confusion matrix of entropy-best is:")
print(bn_matrix)

# second tree with gini
MyDT=DecisionTreeClassifier(criterion='gini',
                            splitter='best',
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=None,
                            random_state=None,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None,
                            class_weight=None)

MyDT.fit(TrainDF, TrainLabels)
feature_names=TrainDF.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                                   feature_names=feature_names,
                                   class_names=topics,
                                   filled=True, rounded=True,
                                   special_characters=True)

graph = graphviz.Source(Tree_Object)
graph.render("Mytree2")

## Confusion Matrix

DT_pred = MyDT.predict(TestDF)
bn_matrix = confusion_matrix(TestLabels, DT_pred)
print("\nThe confusion matrix gini-best is:")
print(bn_matrix)

# third tree with different training size
TrainDF, TestDF = train_test_split(df, test_size=0.15)

TestLabels = TestDF["LABEL"]
TestDF = TestDF.drop(["LABEL"], axis=1)
TrainLabels = TrainDF["LABEL"]
TrainDF = TrainDF.drop(["LABEL"], axis=1)

MyDT=DecisionTreeClassifier(criterion='entropy',
                            splitter='best',
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=None,
                            random_state=None,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None,
                            class_weight=None)

MyDT.fit(TrainDF, TrainLabels)
feature_names=TrainDF.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                                   feature_names=feature_names,
                                   class_names=topics,
                                   filled=True, rounded=True,
                                   special_characters=True)

graph = graphviz.Source(Tree_Object)
graph.render("Mytree3")

## Confusion Matrix
DT_pred = MyDT.predict(TestDF)
bn_matrix = confusion_matrix(TestLabels, DT_pred)
print("\nThe confusion matrix of more training size is:")
print(bn_matrix)


