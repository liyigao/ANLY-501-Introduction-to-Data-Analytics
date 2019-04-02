# -*- coding: utf-8 -*-
"""
Created on Nov 3rd, 2017
Modified by Qixu Cao on Nov 3rd
"""
# Transforming Data and Clustering Examples

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add variables to twitter dataset:
# Add tweets character lengths ("text_length") (for detecting outliers)
# Add sentiment word counts ("positive_count" and "negative_count")
# Add total number of sentiment words ("total_count")
# Add sentiment score ("score"="positive_count"-"negative_count")
def addVariables(dataframe, stopwords):
    with open("positive_words.txt", "r") as Pfile:
        Ptext = Pfile.read()
        positiveList = Ptext.split()
    with open("negative_words.txt", "r") as Nfile:
        Ntext = Nfile.read()
        negativeList = Ntext.split()
    dataframe["character_length"] = dataframe["text"].apply(lambda x: len(x))
    dataframe["positive_count"] = dataframe["text"].apply(lambda x: wordCount(x, positiveList, stopwords))
    dataframe["negative_count"] = dataframe["text"].apply(lambda x: wordCount(x, negativeList, stopwords))
    dataframe["total_count"] = dataframe["positive_count"] + dataframe["negative_count"]
    dataframe["score"] = dataframe["positive_count"] - dataframe["negative_count"]
    # dataframe.to_csv("complete_twitter_data.csv", index = False)
    return dataframe

def binStock(dataframe):
    # myData = pd.read_csv('clean_stock_data.csv' , sep=',', encoding='latin1')
    priceBinNames = range(1,5)
    volumeBinNames = range(1,7)
    priceBins=[0, 10, 100, 1000, 10000]
    volumeBins=[0, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    myData['priceRange'] = pd.cut(myData["4. close"], priceBins, labels=priceBinNames)
    myData['volumeRange'] = pd.cut(myData["5. volume"], volumeBins, labels=volumeBinNames)
    # print(max(myData["4. close"]))
    # print(max(myData["5. volume"]))
    print(myData.head())
    myData.to_csv("final_stock_data.csv", sep=',', encoding='utf-8')

def binTwitter(dataframe):
    # myData = pd.read_csv('complete_twitter_data_2.csv' , sep=',', encoding='latin1')
    names = range(1,6)
    followerNumberBins=[-1, 1000, 10000, 100000, 1000000, 10000000]
    myData['userFollowersNumber'] = pd.cut(myData['user_followers'], followerNumberBins, labels=names)
    # print(max(myData['character_length']))
    print(myData.head())
    myData.to_csv("final_twitter_data.csv", sep=',', encoding='utf-8')

def histogram(dataset):
    # subset = dataset['priceRange']
    subset = dataset['volumeRange']
    subset.hist()
    # plt.title("Distribution of twitter followers volumn")
    plt.title("Distribution of stock volume range")
    plt.show()

if __name__ == '__main__':
    twitterdata = pd.read_csv("clean_twitter_data.csv", encoding='latin1')
    # Count positive/negative words
    twitterdata_new = addVariables(twitterdata, stop)
    # Binning twitter data
    binTwitter(twitterdata_new)
    stockdata = pd.read_csv('clean_stock_data.csv' , sep=',', encoding='latin1')
    # Binning stock data
    binStock(stockdata)
    # Plot histograms
    dataset=pd.read_csv('final_stock_data.csv', sep=',', encoding='latin1')
    histogram(dataset)
