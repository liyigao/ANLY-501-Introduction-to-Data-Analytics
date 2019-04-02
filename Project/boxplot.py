# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:48:43 2017

@author: Yigao
"""

import matplotlib.pyplot as plt
import pandas as pd

## Boxplots
def getBoxplots():
    stockdata = pd.read_csv("clean_stock_data.csv")
    twitterdata = pd.read_csv("complete_twitter_data.csv", encoding = 'latin1')
    stockboxplot = stockdata[["1. open", "2. high", "3. low", "4. close", "5. volume"]]
    stockboxplot.columns = ["open", "high", "low", "close", "volume"]
    twitterboxplot = twitterdata[["user_followers", "character_length", "positive_count", "negative_count", "total_count", "score"]]
    plt.figure()
    stockboxplot.boxplot(column = "open")
    plt.figure()
    stockboxplot.boxplot(column = "high")
    plt.figure()
    stockboxplot.boxplot(column = "low")
    plt.figure()
    stockboxplot.boxplot(column = "close")
    plt.figure()
    stockboxplot.boxplot(column = "volume")
    plt.figure()
    twitterboxplot.boxplot(column = "user_followers")
    plt.figure()
    twitterboxplot.boxplot(column = "character_length")
    plt.figure()
    twitterboxplot.boxplot(column = "positive_count")
    plt.figure()
    twitterboxplot.boxplot(column = "negative_count")
    plt.figure()
    twitterboxplot.boxplot(column = "total_count")
    plt.figure()
    twitterboxplot.boxplot(column = "score")
    plt.figure()
    
def main():
    getBoxplots()

main()
