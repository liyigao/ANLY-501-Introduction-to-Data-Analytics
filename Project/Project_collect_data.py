# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:30:46 2017

@author: Yigao Li
@author: Qixu Cao
@author: Taoran Yu
@author: Xiaoman Dong

"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv
from datetime import datetime
import requests
import pandas as pd
from twython import Twython
import json
import time



## Get S&P500 stock ticker by parsing the wikipedia page
def getSP500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    page = urlopen(url)
    soup = BeautifulSoup(page, "lxml")
    table = soup.find('table', attrs={'class':"wikitable sortable"})
    rows = table.find_all('tr')
    #Ticker = set()
    Ticker = list()
    i = 0
    for row in rows:
        if (row.find('td')):
            ticker = row.find('td').string
            i = i + 1
            print("Added the %d ticker: %s\n", (i, ticker))
            #Ticker.add(ticker)
            Ticker.append(ticker)
            with open("tickerList.txt", "a") as file:
                file.write(ticker)
                file.write("\n")
    print("Added %d tickers", (i))
    return Ticker

## Get hourly stock price via AlphaVantage free API
def getStockPrice (symbolList):
    outputdf = pd.DataFrame()
    for symbol in symbolList:
        print("Getting stock price for", symbol)
        BaseURL = "https://www.alphavantage.co/query"
        URLPost = {'function' : 'TIME_SERIES_INTRADAY',
                   'symbol' : symbol,
                   'interval' : '60min',
                   'outputsize' : 'full',
                   'apikey' : '**************'} # api key hide on purpose
        response = requests.get(BaseURL, URLPost)
        jsontxt = response.json()
        jsondict = jsontxt['Time Series (60min)'] ## Inner dictionary
        jsondf = pd.DataFrame(jsondict).transpose()
        jsondf['symbol'] = pd.Series(symbol, index = jsondf.index)
        outputdf = outputdf.append(jsondf)
        time.sleep(1) # AlphaVantage has API Rate Limit, 100 calls per minute
        print("Generating csv file....")
        outputdf.to_csv("S&P500_stock_price.csv")
    return outputdf

## Get Twitter Data, this function will run 5 times, filename is default, can be changed for test purpose
def getTwitterData (symbolList, filename = "S_P500_twitter.csv"):
    TWITTER_APP_KEY = '****************' ##API key hide on purpose
    TWITTER_APP_KEY_SECRET = '**********************************************'
    TWITTER_ACCESS_TOKEN = '**************************************************'
    TWITTER_ACCESS_TOKEN_SECRET = '*********************************************'
    t = Twython(app_key=TWITTER_APP_KEY,
                app_secret=TWITTER_APP_KEY_SECRET,
                oauth_token=TWITTER_ACCESS_TOKEN,
                oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

    # Read a file to get a start location in symbolList, along with end location
    start = readCounter()
    increment = int(len(symbolList)/5)
    end = start + increment
    twitterdata = pd.DataFrame()
    for ticker in symbolList[start:end]:
        hashticker = "#"+ticker
        search = t.search(q=hashticker, ##stock code and number of twitters you want
                          count=100)
        tweets = search['statuses']
        column_list = ['ticker', 'user_id', 'user_name', 'user_followers', 'user_verified', 'user_location', 'user_description', 'text', 'created_at', 'source']
        df = pd.DataFrame(columns=column_list)
        start = start + 1
        print("Processing the", start, "ticker:", ticker, "\n")
        #print("Processing ....", (ticker))
        for tweet in tweets:
            df = df.append(
                {
                    'ticker':   ticker,
                    'user_id':  tweet['user']['id'],
                    'user_name':     tweet['user']['name'],
                    'user_followers':tweet['user']['followers_count'],
                    'user_verified': tweet['user']['verified'],
                    'user_location': tweet['user']['location'],
                    'user_description': tweet['user']['description'],
                    'text': tweet['text'],
                    'created_at': tweet['created_at'],
                    'source': tweet['source']
                },
                ignore_index = True
            )
        twitterdata = twitterdata.append(df)
        # print (tweet['user']['id'],'\n',tweet['user']['name'],'\n',
        #        tweet['user']['verified'],'\n',tweet['user']['location'],'\n',tweet['user']['description'], '\n',
        #        tweet['text'], '\n',tweet['created_at'],'\n',tweet['source'],'\n\n\n')
    if end == increment:    # 1st time write csv with header
        twitterdata.to_csv(filename, header = True, mode = "a")
    else:                   # append csv without header
        twitterdata.to_csv(filename, header = False, mode = "a")
    
    if end != int(len(symbolList)): # store end location to file for next run
        deletePreviousCounter()
        writeCurrentCounter(end)
    
    #print(twitterdata)
    return twitterdata

    #
    # import pandas as pd
    # df = pd.DataFrame()
    # df = df.append({'foo':1, 'bar':2}, ignore_index=True)

# Create file for storing and reading ticker location
def createInitialCounter():
    with open("count.txt", "w") as counter:
        counter.write("0")

# Read location in ticker list to start collecting data
def readCounter():
    with open("count.txt", "r") as counter:
        return int(counter.read())

# Delete old count
def deletePreviousCounter():
    with open("count.txt", "w") as counter:
        counter.close()

# Store a new count for next run
def writeCurrentCounter(count):
    with open("count.txt", "w") as counter:
        counter.write(str(count))
        
if __name__ == "__main__":
    #createInitialCounter()
    tickerList = getSP500()
    #testList = tickerList[:50]
    #print(tickerList)

    # Because Twitter has Rate Limit for API, ticker list is split into 5 parts and time interval between 2 runs is 15min.
    # For testing purpose, run the code below:
    #getTwitterData(tickerList, filename = "test_Twitter.csv")
    #getStockPrice(tickerList)
    #for i in range(5):
    #    getTwitterData(tickerList)
    #    time.sleep(901)
