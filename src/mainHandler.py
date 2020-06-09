import csv
import nltk
from nltk.tokenize import word_tokenize
# nltk.download() #if stopwords not downloaded: Coropora -> Stopwords -> download
stopwordsDe = nltk.corpus.stopwords.words('german')
stopwordsEn = nltk.corpus.stopwords.words('english')

allTweets = []

class Tweet:
  def __init__(self, tweet, hatespeech):
    self.tweet = tweet
    self.hatespeech = hatespeech

def loadGermanCsvToTweetObject():
  with open('C:/Users/flavi/Desktop/hateSpeechAnalysis-master/germanTweets.csv', encoding="utf8", newline='') as csvfile:
      tweetreader = csv.reader(csvfile, delimiter=',')
      for row in tweetreader:
          tweetText = row[0]
          item = Tweet(tweetText, False)
          allTweets.append(item)

def loadEnglishCsvToTweetObject():   
  with open('C:/Users/flavi/Desktop/hateSpeechAnalysis-master/englishTweets.csv', encoding="utf8", newline='') as csvfile:
      tweetreader = csv.reader(csvfile, delimiter=';')
      for row in tweetreader:
          tweetText = row[1]
          item = Tweet(tweetText, False)
          allTweets.append(item)

def removeStopWords(stopwords): 
  for tweet in allTweets:
    text = tweet.tweet
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stopwords] 
    filtered_sentence = [] 
    for w in word_tokens: 
      if w not in stopwords: 
          filtered_sentence.append(w) 
    print(word_tokens) 
    print(filtered_sentence) 


#loadEnglishCsvToTweetObject()
loadGermanCsvToTweetObject()
removeStopWords(stopwordsDe)
