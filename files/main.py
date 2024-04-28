import requests
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

End = "https://newsapi.org/v1/articles"

# set the taget word in title of articles
url = ('https://newsapi.org/v2/everything?'
       'qInTitle=marvel%20character%20OR%20marvel%20universe&'
       'language=en&'
       'pageSize=100&'
       'apiKey=378458c93874414789c0f888a0814170'
       )
print(url)
response = requests.get(url)
jsontxt = response.json()
print(jsontxt, "\n")

# get a new csv file to write
filename="newsapi.csv"
MyFILE=open(filename,"w")

WriteThis="Date,Source,Title,Headline\n"
MyFILE.write(WriteThis)
MyFILE.close()

## Open the file for append
MyFILE = open(filename, "a")

# check and clean the style of data
for items in jsontxt["articles"]:
       print(items, "\n\n\n")

       Source = items["source"]["id"]
       print(Source)

       Date = items["publishedAt"]
       ##clean up the date
       NewDate = Date.split("T")
       Date = NewDate[0]
       print(Date)

       # clean the Title
       Title=items["title"]
       Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', Title, flags=re.IGNORECASE)
       Title=re.sub(' +', ' ', Title, flags=re.IGNORECASE)
       Title=re.sub(r'\"', ' ', Title, flags=re.IGNORECASE)
       Title = re.sub(r'[^a-zA-Z]', " ", Title, flags=re.VERBOSE)
       Title = Title.replace(',', '')
       Title = ' '.join(Title.split())
       Title = re.sub("\n|\r", "", Title)

       # Clean the headline
       Headline = items["description"]
       Headline = re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
       Headline = re.sub(' +', ' ', Headline, flags=re.IGNORECASE)
       Headline = re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
       Headline = re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
       Headline = Headline.replace(',', '')
       Headline = ' '.join(Headline.split())
       Headline = re.sub("\n|\r", "", Headline)

       Headline = ' '.join([wd for wd in Headline.split() if len(wd) > 3])

       print(Title)
       print(Headline)

       # write information into the csv file we created before
       WriteThis = str(Date) + "," + str(Source) + "," + str(Title) + "," + str(Headline) + "\n"
       MyFILE.write(WriteThis)

MyFILE.close()



