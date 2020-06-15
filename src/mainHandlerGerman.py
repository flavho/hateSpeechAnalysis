import csv
import nltk
import os
import html
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim 
from sklearn.pipeline import Pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#nltk.download() #if stopwords not downloaded: Coropora -> Stopwords -> download
stopwordsDe = nltk.corpus.stopwords.words('german')
myPath = os.path.abspath(os.path.dirname(__file__))
df = ""


def loadGermanCsvToTweetObject():
    path = os.path.join(myPath, "..\data\germanTweets.csv")
    df = pd.read_csv(path)
    df.drop(columns=['HatespeechOrNot (Expert 1)','HatespeechOrNot (Expert 2)'], axis=1, inplace=True)
    return df
    


def removeStopWordsAndTokenize(stopwords, tweetText):
    # after calling this function it will put the tokenized Text without Stopwords in the fitting Tweet Object
    word_tokens = word_tokenize(tweetText)
    filtered_sentence = [w for w in word_tokens if not w in stopwords]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stopwords:
            filtered_sentence.append(w)
    return TreebankWordDetokenizer().detokenize(filtered_sentence)


def lowerString(tweetText):
    # all Strings to lower case
    return tweetText.lower()


def convertHtml(tweetText):
    # Convert html to Unicode
    return html.unescape(tweetText)


def adressParsing(tweetText):
    # wwww and https to ""
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweetText)


def usernameToEmptyString(tweetText):
    # username to emtpy String
    return re.sub('@[^\s]+', '', tweetText)


def removeWhiteSpaces(tweetText):
    # Remove all white spaces
    return re.sub('[\s]+', ' ', tweetText)


def removeHashtags(tweetText):
    # Replace # with ""
    return re.sub(r'#([^\s]+)', r'\1', tweetText)


def trimString(tweetText):
    # removes Spaces at start / end
    return tweetText.strip('\'"')


def removeNonChars(tweetText):
    # remove all characters that are not in the alphabet
    return re.sub('[^A-Za-z0-9 ]+', '', tweetText)


def formatAllTweetsforNltkDe(train):
    # takes all Tweets from csv and puts them in Array after they got formated and tokenized
    for index, row in train.iterrows():
      tweetText = row['Tweet']
      newText = lowerString(convertHtml(adressParsing(usernameToEmptyString(removeWhiteSpaces(removeHashtags(trimString(removeNonChars(tweetText))))))))
      edtitedWoStopWords = removeStopWordsAndTokenize(stopwordsDe,newText)
      row['Tweet'] = edtitedWoStopWords
      #print(edtitedWoStopWords)

def splitData(df):
    return train_test_split(df["Tweet"], df["Hatespeech"], test_size=0.25, random_state=42)




def getMostCommonWords(model):
    words = list(model.wv.vocab)
    return words

def createModel(X_train_vectorized,y_train):
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    return model

def tfidVectorizeCommonWords(X_train, vect):
    X_train_vectorized = vect.transform(X_train)
    #print(X_train_vectorized)
    return X_train_vectorized

def main():
    df = loadGermanCsvToTweetObject()
    formatAllTweetsforNltkDe(df)
    X_train, X_test, y_train, y_test = splitData(df)
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = tfidVectorizeCommonWords(X_train, vect)
    #print(X_train_vectorized)
    model = createModel(X_train_vectorized,y_train)
    #print(model)
    feature_names = np.array(vect.get_feature_names())
    #print(feature_names)
    sorted_tfidf_index = model.coef_[0].argsort()
    #print(sorted_tfidf_index)
    predictions = model.predict(vect.transform(X_test))
    print(predictions)
    y_test = np.array(y_test)
    print(y_test)
    print(roc_auc_score(y_test, predictions))



main()
  
