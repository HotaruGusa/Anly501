# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
import re

import warnings
warnings.filterwarnings('ignore')

FileNameList=os.listdir("corpus")
#print(type(FileNameList))
#print(FileNameList)

# create two blank list
MyFileNameList=[]
FileNames=[]

#for nextfile in os.listdir("corpus"):
#    print(nextfile)

# add file name to the list
for nextfile in os.listdir("corpus"):
    fullpath="corpus"+"/"+nextfile
    #print(fullpath)
    MyFileNameList.append(fullpath)
    FileNames.append(nextfile)

# since the data are not cleaned yet, use basic function to clean each one
for files in MyFileNameList:
    with open(files, "r") as f:
        content = f.read()
        content = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', content, flags=re.MULTILINE)
        content = re.sub(r'[,.;@#?!&$\-\']+', ' ', content, flags=re.IGNORECASE)
        content = re.sub(' +', ' ', content, flags=re.IGNORECASE)
        content = re.sub(r'\"', ' ', content, flags=re.IGNORECASE)
        content = re.sub(r'[^a-zA-Z]', " ", content, flags=re.VERBOSE)
        content = content.replace(',', '')
        content = ' '.join(content.split())
        content = re.sub("\n|\r", "", content)
    # write the cleaned text data back
    file = open(files, "w")
    file.write(content)

#print(MyFileNameList)
#print(FileNames)

# using CountVectorizer to create a document term matrix
MyCV=CountVectorizer(input='filename',stop_words='english')
My_DTM=MyCV.fit_transform(MyFileNameList)

MyColumnNames=MyCV.get_feature_names()
#print("The vocab is: ", MyColumnNames, "\n\n")

My_DF=pd.DataFrame(My_DTM.toarray(),columns=MyColumnNames)

#print(FileNames)

# create labels
CleanNames=[]
for filename in FileNames:
    #print(type(filename))
    ## remove the number and the .txt from each file name
    newName=filename.split(".")
    CleanNames.append(newName[0])

#print(CleanNames)

# write labels to the first of docs
My_DF.insert(0, 'LABEL', CleanNames)
print(My_DF)
# write to a csv file
My_DF.to_csv('corpus_df.csv', index=False)

