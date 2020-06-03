import csv


allTweets = []

class Tweet:
  def __init__(self, tweet, hatespeech):
    self.tweet = tweet
    self.hatespeech = hatespeech

def loadCsvToTweetObject():
    with open('C:/Users/flavi/Desktop/NLP/germanTweets.csv', newline='') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter=',')
        for row in tweetreader:
            tweetText = row[0]
            item = Tweet(tweetText, False)
            allTweets.append(item)
            
            

loadCsvToTweetObject()