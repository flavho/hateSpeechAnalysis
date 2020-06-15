import csv
import nltk
import os
import html
import re
import tkinter as tk  
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
#classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
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
    #return filtered_sentence


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


def formatAllTweetsforNltkDe(df):
    # takes all Tweets from csv and puts them in Array after they got formated and tokenized
    dfInput = { 'Tweet': [],'Hatespeech': [] }
    for index, row in df.iterrows():
        tweetText = row['Tweet']
        hatespeechIndicator = row['Hatespeech']
        newText = lowerString(convertHtml(adressParsing(usernameToEmptyString(removeWhiteSpaces(removeHashtags(trimString(removeNonChars(tweetText))))))))
        edtitedWoStopWords = removeStopWordsAndTokenize(stopwordsDe,newText)
        dfInput["Tweet"].append(edtitedWoStopWords)
        dfInput["Hatespeech"].append(hatespeechIndicator)
       
    clearedDf = pd.DataFrame(dfInput, columns = ['Tweet', 'Hatespeech'])
    return clearedDf



def splitData(df):
    return train_test_split(df["Tweet"], df["Hatespeech"], test_size=0.3, random_state=42)



def tfidVectorizeCommonWords(X_train, vect):
    X_train_vectorized = vect.transform(X_train)
    #print(X_train_vectorized)
    return X_train_vectorized

def logRegreAndTfidfVectorizer():
    iniDf = loadGermanCsvToTweetObject()
    df = formatAllTweetsforNltkDe(iniDf)
    #split Dataset
    X_train, X_test, y_train, y_test = splitData(df)
    #Tfid Vectorizer
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = tfidVectorizeCommonWords(X_train, vect)
    #create Model
    model = LogisticRegression(C=5.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=20000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.01, verbose=0,
                   warm_start=False)
    model.fit(X_train_vectorized, y_train)
    sorted_tfidf_index = model.coef_[0].argsort()
    #predict for the Testdataset
    predictions = model.predict(vect.transform(X_test))
    print(predictions)
    y_test = np.array(y_test)
    xTestTweets = np.array(X_test)
    #Print Tweet texts and how they are predictet
    #counter = 0
    #for hate in predictions: 
    #    if(hate==0):
    #        print("No Hate Speech:")
    #       print(xTestTweets[counter])
    #    else: 
    #        print("Hatespeech detected in:")
    #       print(xTestTweets[counter])
    #    counter = counter + 1

    print(f"Accuracy is of Logreg and Tfidf is: {roc_auc_score(y_test, predictions)}")

def supVecMacAndTfidfVectorizer():
    iniDf = loadGermanCsvToTweetObject()
    df = formatAllTweetsforNltkDe(iniDf)
    X_train, X_test, y_train, y_test = splitData(df)
    vect = TfidfVectorizer().fit(X_train)

    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    x_len = X_train.apply(len)
    X_train_aug = add_feature(X_train_vectorized, x_len)
    
    x_len2 = X_test.apply(len)
    X_test_aug = add_feature(X_test_vectorized, x_len2)
    
    model = SVC(C=20000, max_iter=10).fit(X_train_aug, y_train)
    predictions = model.predict(X_test_aug)
    print(predictions)

    roc = roc_auc_score(y_test, predictions)
    print(f"Accuracy is of supVec and Tfidf is: {roc}")

def add_feature(X, feature_to_add):
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def multiNaiveBayesAndTfidfVectorizer():
    iniDf = loadGermanCsvToTweetObject()
    df = formatAllTweetsforNltkDe(iniDf)
    X_train, X_test, y_train, y_test = splitData(df)
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = tfidVectorizeCommonWords(X_train, vect)
    
    model = MultinomialNB(alpha=0.1,fit_prior=True)
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(vect.transform(X_test))
    print(predictions)
    roc = roc_auc_score(y_test, predictions)
    print(f"Accuracy is of MultiNaive Bayes and Tfidf is: {roc}")

def createGui():
    root = tk.Tk()
    firstButton = tk.Button(root, text="Support Vector Machine & TfidfVectorizer", command=supVecMacAndTfidfVectorizer)
    firstButton.pack()
    secondButton = tk.Button(root, text="Logistic Regression & TfidfVectorizer", command=logRegreAndTfidfVectorizer)
    secondButton.pack()
    thirdButton = tk.Button(root, text="Multi-Nominal Naive Bayes & TfidfVectorizer", command=multiNaiveBayesAndTfidfVectorizer)
    thirdButton.pack()
    root.mainloop()

def main():
    createGui()
       

    

    


main()
  
