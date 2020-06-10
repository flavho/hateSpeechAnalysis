import csv
import nltk
import os 
from nltk.tokenize import word_tokenize
# nltk.download() #if stopwords not downloaded: Coropora -> Stopwords -> download
stopwordsDe = nltk.corpus.stopwords.words('german')
stopwordsEn = nltk.corpus.stopwords.words('english')
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

def loadEnglishCsvToTweetObject():   
  path = os.path.join(myPath, "..\data\englishTweets.csv")
  with open(path, encoding="utf8", newline='') as csvfile:
      tweetreader = csv.reader(csvfile, delimiter=';')
      for row in tweetreader:
          tweetText = row[1]
          item = Tweet(tweetText, False)
          allTweets.append(item)

def removeStopWords(stopwords): 
  #after calling this function it will put the tokenized Text without Stopwords in the fitting Tweet Object
  for i in range(len(allTweets)):
    text = allTweets[i].tweet
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stopwords] 
    filtered_sentence = [] 
    for w in word_tokens: 
      if w not in stopwords: 
          filtered_sentence.append(w)
    allTweets[i].tweet = filtered_sentence


    
#loadEnglishCsvToTweetObject()
loadGermanCsvToTweetObject()
removeStopWords(stopwordsDe)
#removeStopWords(stopwordsEn)
