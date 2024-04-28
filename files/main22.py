import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_DF=pd.read_csv("newsapi.csv")
#print(CSV_DF)


My_Content_List=[]
My_Labels_List=[]

with open("newsapi.csv", "r") as My_FILE:
    next(My_FILE)  ## skip the first row

    for next_row in My_FILE:
        Row_Elements=next_row.split(",")
        My_Labels_List.append(Row_Elements[0])
        # Add title and content to the content list
        My_Content_List.append(Row_Elements[2]+Row_Elements[3])

# using CountVectorizer to create a document term matrix
MyCV_content=CountVectorizer(input='content',
                        stop_words='english',
                        )
My_DTM2=MyCV_content.fit_transform(My_Content_List)

# write labels to the first of docs
ColNames=MyCV_content.get_feature_names()
My_DF_content=pd.DataFrame(My_DTM2.toarray(),columns=ColNames)
My_DF_content.insert(0, 'LABEL', My_Labels_List)

# write to a csv file
My_DF_content.to_csv('MyClean_CSV_Data.csv', index=False)



