import csv
import nltk
import os 
import html
import re
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download() #if stopwords not downloaded: Coropora -> Stopwords -> download
stopwordsDe = nltk.corpus.stopwords.words('german')
myPath = os.path.abspath(os.path.dirname(__file__))
allTweets = []
removeStopWordsLoopCounter = 0

class Tweet:
  def __init__(self, tweet, hatespeech):
    self.tweet = tweet
    self.hatespeech = hatespeech

def loadGermanCsvToTweetObject():
  path = os.path.join(myPath, "..\data\germanTweets.csv")
  with open(path, encoding="utf8", newline='') as csvfile:
      tweetreader = csv.reader(csvfile, delimiter=',')
      for row in tweetreader:
          tweetText = row[0]
          item = Tweet(tweetText, False)
          allTweets.append(item)

def removeStopWordsAndTokenize(stopwords, tweetText): 
  #after calling this function it will put the tokenized Text without Stopwords in the fitting Tweet Object
    word_tokens = word_tokenize(tweetText) 
    filtered_sentence = [w for w in word_tokens if not w in stopwords] 
    filtered_sentence = [] 
    for w in word_tokens: 
      if w not in stopwords: 
          filtered_sentence.append(w)
    return filtered_sentence

def lowerString(tweetText):
  # all Strings to lower case
  return tweetText.lower()

def convertHtml(tweetText):
  # Convert html to Unicode
  return html.unescape(tweetText)

def adressParsing(tweetText):
  # wwww and https to ""
  return re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweetText)
    
def usernameToEmptyString(tweetText):
  #username to emtpy String
  return re.sub('@[^\s]+','',tweetText)
    
def removeWhiteSpaces(tweetText):
  #Remove all white spaces
  return re.sub('[\s]+', ' ', tweetText)
  
def removeHashtags(tweetText):
  #Replace # with ""
  return re.sub(r'#([^\s]+)', r'\1', tweetText)

def trimString(tweetText):
  # removes Spaces at start / end
  return tweetText.strip('\'"')

def removeNonChars(tweetText):
  # remove all characters that are not in the alphabet
  return re.sub('[^A-Za-z0-9 ]+','', tweetText)

def formatAllTweetsforNltkDe():
  # takes all Tweets from csv and puts them in Array after they got formated and tokenized
  for i in range(len(allTweets)):
    text = allTweets[i].tweet
    toTokenize = lowerString(convertHtml(adressParsing(usernameToEmptyString(removeWhiteSpaces(removeHashtags(trimString(removeNonChars(text))))))))
    print(removeStopWordsAndTokenize(stopwordsDe, toTokenize))
    allTweets[i].tweet = removeStopWordsAndTokenize(stopwordsDe, toTokenize)
    
    


def main():
  loadGermanCsvToTweetObject()
  formatAllTweetsforNltkDe()

main()
