# ANLY 501 Project 3 Group 1
# Qixu Cao
# Xiaoman Dong
# Yigao Li
# Tao Yu

import re
import plotly
import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
import datetime
import numpy as np
import nltk
from plotly import tools
from collections import defaultdict
from scipy.stats import ttest_ind
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

plotly.tools.set_credentials_file(username='liyigao', api_key='6EmMOoSuy5cfO0f6PLyK')

def getDataset4(stockdata, twitterdata):
    datahourly = stockdata[["Unnamed: 0.1.1", "1. open", "2. high", "3. low", "4. close", "5. volume", "symbol", "priceRange", "volumeRange"]]
    datahourly.columns = ["datetime", "open", "high", "low", "close", "volume", "ticker", "priceRange", "volumeRange"]
    time_modified = []
    for tweetTime in twitterdata["created_at"]:
        tweetTimeElement = datetime.datetime.strptime(tweetTime, '%Y-%m-%d %H:%M:%S')
        tweetTimeElement = tweetTimeElement.replace(minute = 0, second = 0)
        tweetTimeElement = tweetTimeElement + datetime.timedelta(hours = 1)
        time_modified.append(tweetTimeElement.strftime('%Y-%m-%d %H:%M:%S'))
    time_modified_series = pd.Series(time_modified)
    twitterdata["datetime"] = time_modified_series.values
    tdf = twitterdata[["ticker", "positive_count", "negative_count", "total_count", "score", "datetime"]]
    tdf1 = twitterdata[["ticker", "positive_count", "datetime"]]
    tdf1 = tdf1.groupby(["ticker", "datetime"]).sum()
    tdf2 = twitterdata[["ticker", "negative_count", "datetime"]]
    tdf2 = tdf2.groupby(["ticker", "datetime"]).sum()
    tdf_join = pd.concat([tdf1, tdf2], axis = 1).reset_index()
    tdf_join["total_count"] = tdf_join["positive_count"] + tdf_join["negative_count"]
    tdf_join["score"] = tdf_join["positive_count"] - tdf_join["negative_count"]
    dataset4 = datahourly.merge(tdf_join, how = "left", on = ["ticker", "datetime"])
    dataset4 = dataset4.fillna(0)
    dataset4.to_csv("dataset4.csv", index = None)
    return dataset4

# convert letters to lower case
process_String_To_Small_Letters = lambda x: x.lower()
# convert letters to upper case
process_String_To_Capital_Letters = lambda x: x.upper()
# remove strings other than letters, numbers, #, ?, !
process_String_Keep_Only_Letters_Numbers = lambda x: re.sub("[^a-zA-Z0-9#?!]+", " ", str(x))
# remove whitespce
process_String_Remove_Space = lambda x: x.strip()
# remove strings less than 5 characters
process_String_Length_Less_Than_5 = lambda x: " " if len(str(x).strip())<5 else x

def clean(df):
    twitter_df1 = df
    twitter_df1 = twitter_df1.dropna(subset=['text'])
    row_number_data_twitter_df1 = len(twitter_df1.index)
    twitter_df2 = twitter_df1
    twitter_df2['text'] = twitter_df2['text'].apply(process_String_Keep_Only_Letters_Numbers)
    twitter_df2['user_description'] = twitter_df2['user_description'].apply(process_String_Keep_Only_Letters_Numbers)
    twitter_df2['user_location'] = twitter_df2['user_location'].apply(process_String_Keep_Only_Letters_Numbers)
    twitter_df2['text'] = twitter_df2['text'].apply(process_String_Length_Less_Than_5)
    twitter_df2['text'] = twitter_df2['text'].apply(lambda x: x.strip()).replace('', np.nan)
    row_number_data_twitter_df2 = len(twitter_df2.index)
    twitter_df2 = twitter_df2.dropna(subset=['text'])
    row_number_data_twitter_df2 = len(twitter_df2.index)
    twitter_df3 = twitter_df2
    twitter_df3 = twitter_df3.dropna(subset=['created_at'])
    ten_row_data_twitter = twitter_df3.iloc[:10]
    twitter_df3['text'] = twitter_df3['text'].apply(process_String_To_Small_Letters)
    return twitter_df3

# - A function that returns frequency of each word occurrence in the file with filename as a dictionay
def getWordsFrequency(filename):
    document_text = open(filename, encoding='utf-8')
    text_string = document_text.read().lower()
    document_text.close()
    
    text_string = text_string.translate ({ord(c): " " for c in "!’@#$%^&*()[]{};:,./<>—?\-|'`~-=_+\""})
    words=text_string.split()
    frequency = { }
    for word in words:
        count = frequency.get(word,0)
        frequency[word] = count + 1
    return frequency

# -  A function that returns frequency of each multiwords in the with the filename as a dictionary
def getMultiwordUnitsFrequency(filename):
    document_text = open(filename, encoding = 'utf-8')      # open test file
    text_string = document_text.read().lower()
    document_text.close()

    multiwords_text = open("multiwords.txt", 'r')           # open multiwords file
    multiwords = multiwords_text.readlines()
    multiwords_text.close()

    frequency = {}                                          # create dictionary and count frequency of each multiwords in test file
    for word in multiwords:
        word = word.strip()
        frequency[word] = text_string.count(word.lower())
    return frequency

# - A function that returns sorted List Of Keys Based On their values in descending order
def sortListOfKeysBasedOnFreq(frequenciesDict):
    return sorted(frequenciesDict,key=frequenciesDict.get,reverse=True)

# - Mix words and MultiWords and limit the number of items in the final output
def mixWordsAndMultiWords(wordsFrequency,multiWordsFrequency, stopWords, maxNumOfWords):
    # Assume the output is the wordsFrequency dictonary without the words in the stopWords
    frequency = {}                      # create dictionary of all frequency of single words which are not in stopWords
    for w,f in wordsFrequency.items():
        if w not in stopWords:
            frequency[w] = f

    # Keep adding multiwords to the final result as long as the 10th (maxNumOfWords) highest frequency is lower than next max frequency
    sortedMultiWordsBasedOnFreq = sortListOfKeysBasedOnFreq(multiWordsFrequency)
    for mw in sortedMultiWordsBasedOnFreq:
        sorted_frequency = sortListOfKeysBasedOnFreq(frequency)
        if frequency[sorted_frequency[maxNumOfWords - 1]] > multiWordsFrequency[mw]:        # terminate when frequency of multiword is not large enough
            break
        lowesPossibleFrequency = frequency[sorted_frequency[maxNumOfWords - 1]]
        if multiWordsFrequency[mw] < lowesPossibleFrequency:
            break
        for word in mw.split():
            if word not in stopWords:
                frequency[word.lower()] = frequency[word.lower()] - multiWordsFrequency[mw]
        frequency[mw] = multiWordsFrequency[mw]                                             # append new multiword with frequency in the dictionary

    # Limit number of output
    limitedFrequency = {}
    i = 0
    for w in sorted_frequency:              # limit the number of words shown on WordCloud
        limitedFrequency[w] = frequency[w]
        i = i + 1
        if i == maxNumOfWords:
            break

    return limitedFrequency

# Count number of words appeared from a text in a given wordlist
def wordCount(text, wordlist, stop):
    tokens = nltk.word_tokenize(text)
    count = 0
    for word in tokens:
        if word not in stop:
            if word in wordlist:
                count = count + 1
    return count

# A function that returns stop words in a list
def getStopWords():
    document_text = open('stopwords.txt', encoding='utf-8')
    text_string = document_text.read().lower()
    words=text_string.split()
    return words

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
    dataframe.to_csv("complete_twitter_data.csv", index = False)
    return dataframe

def getDataset5(stockdata, twitterdata):
    d110 = stockdata[stockdata.datetime == "2017-11-27 10:00:00"]
    d111 = stockdata[stockdata.datetime == "2017-11-27 11:00:00"]
    d112 = stockdata[stockdata.datetime == "2017-11-27 12:00:00"]
    d113 = stockdata[stockdata.datetime == "2017-11-27 13:00:00"]
    d114 = stockdata[stockdata.datetime == "2017-11-27 14:00:00"]
    d115 = stockdata[stockdata.datetime == "2017-11-27 15:00:00"]
    d116 = stockdata[stockdata.datetime == "2017-11-27 16:00:00"]
    d210 = stockdata[stockdata.datetime == "2017-11-28 10:00:00"]
    d211 = stockdata[stockdata.datetime == "2017-11-28 11:00:00"]
    d212 = stockdata[stockdata.datetime == "2017-11-28 12:00:00"]
    d213 = stockdata[stockdata.datetime == "2017-11-28 13:00:00"]
    d214 = stockdata[stockdata.datetime == "2017-11-28 14:00:00"]
    d215 = stockdata[stockdata.datetime == "2017-11-28 15:00:00"]
    d216 = stockdata[stockdata.datetime == "2017-11-28 16:00:00"]
    dailystock = d110.append(d111, ignore_index = True)
    dailystock = dailystock.append(d112, ignore_index = True)
    dailystock = dailystock.append(d113, ignore_index = True)
    dailystock = dailystock.append(d114, ignore_index = True)
    dailystock = dailystock.append(d115, ignore_index = True)
    dailystock = dailystock.append(d116, ignore_index = True)
    dailystock = dailystock.append(d210, ignore_index = True)
    dailystock = dailystock.append(d211, ignore_index = True)
    dailystock = dailystock.append(d212, ignore_index = True)
    dailystock = dailystock.append(d213, ignore_index = True)
    dailystock = dailystock.append(d214, ignore_index = True)
    dailystock = dailystock.append(d215, ignore_index = True)
    dailystock = dailystock.append(d216, ignore_index = True)
    dailystock = dailystock.sort_values(["ticker", "datetime"])
    dailystock.to_csv("temp1.csv", index = None)
    dailystock = pd.read_csv("temp1.csv")
    databysymbol = pd.DataFrame()
    for i in range(0,3):
        symbol = dailystock["ticker"][14*i]
        d1return = (dailystock.close[14*i+6] - dailystock.open[14*i])/dailystock.open[14*i]
        d2return = (dailystock.close[14*i+13] - dailystock.open[14*i+7])/dailystock.open[14*i+7]
        if d2return >= 0:
            label = "Positive"
        else:
            label = "Negative"
        d1volume = sum(dailystock.volume[14*i:14*i+7])
        d2volume = sum(dailystock.volume[14*i+7:14*i+14])
        volumeChange = d2volume/d1volume - 1
        temp = pd.DataFrame({"ticker":symbol, "d1return":d1return, "d2return":d2return, "%volume":volumeChange, "label":label}, index = [i])
        databysymbol = pd.concat([databysymbol, temp])
    dateonly = []
    for tweetTime in twitterdata["created_at"]:
        tweetTimeElement = datetime.datetime.strptime(tweetTime, '%Y-%m-%d %H:%M:%S')
        dateonly.append(tweetTimeElement.strftime('%Y-%m-%d'))
    dateonly_series = pd.Series(dateonly)
    twitterdata["date"] = dateonly_series.values
    scoreList = []
    for symbol in databysymbol["ticker"]:
        scoreList.append(twitterdata.loc[(twitterdata["date"] == "2017-11-27") & (twitterdata["ticker"] == symbol), "score"].sum())
    scoreList_series = pd.Series(scoreList)
    databysymbol["score"] = scoreList_series.values
    databysymbol.to_csv("dataset5.csv", index = None)
    return databysymbol

def getDataset6(stockdata, twitterdata):
    d110 = stockdata[stockdata.datetime == "2017-11-28 10:00:00"]
    d111 = stockdata[stockdata.datetime == "2017-11-28 11:00:00"]
    d112 = stockdata[stockdata.datetime == "2017-11-28 12:00:00"]
    d113 = stockdata[stockdata.datetime == "2017-11-28 13:00:00"]
    d114 = stockdata[stockdata.datetime == "2017-11-28 14:00:00"]
    d115 = stockdata[stockdata.datetime == "2017-11-28 15:00:00"]
    d116 = stockdata[stockdata.datetime == "2017-11-28 16:00:00"]
    d210 = stockdata[stockdata.datetime == "2017-11-29 10:00:00"]
    d211 = stockdata[stockdata.datetime == "2017-11-29 11:00:00"]
    d212 = stockdata[stockdata.datetime == "2017-11-29 12:00:00"]
    d213 = stockdata[stockdata.datetime == "2017-11-29 13:00:00"]
    d214 = stockdata[stockdata.datetime == "2017-11-29 14:00:00"]
    d215 = stockdata[stockdata.datetime == "2017-11-29 15:00:00"]
    d216 = stockdata[stockdata.datetime == "2017-11-29 16:00:00"]
    dailystock = d110.append(d111, ignore_index = True)
    dailystock = dailystock.append(d112, ignore_index = True)
    dailystock = dailystock.append(d113, ignore_index = True)
    dailystock = dailystock.append(d114, ignore_index = True)
    dailystock = dailystock.append(d115, ignore_index = True)
    dailystock = dailystock.append(d116, ignore_index = True)
    dailystock = dailystock.append(d210, ignore_index = True)
    dailystock = dailystock.append(d211, ignore_index = True)
    dailystock = dailystock.append(d212, ignore_index = True)
    dailystock = dailystock.append(d213, ignore_index = True)
    dailystock = dailystock.append(d214, ignore_index = True)
    dailystock = dailystock.append(d215, ignore_index = True)
    dailystock = dailystock.append(d216, ignore_index = True)
    dailystock = dailystock.sort_values(["ticker", "datetime"])
    dailystock.to_csv("temp1.csv", index = None)
    dailystock = pd.read_csv("temp1.csv")
    databysymbol = pd.DataFrame()
    for i in range(0,3):
        symbol = dailystock["ticker"][14*i]
        d1return = (dailystock.close[14*i+6] - dailystock.open[14*i])/dailystock.open[14*i]
        d2return = (dailystock.close[14*i+13] - dailystock.open[14*i+7])/dailystock.open[14*i+7]
        if d2return >= 0:
            label = "Positive"
        else:
            label = "Negative"
        d1volume = sum(dailystock.volume[14*i:14*i+7])
        d2volume = sum(dailystock.volume[14*i+7:14*i+14])
        volumeChange = d2volume/d1volume - 1
        temp = pd.DataFrame({"ticker":symbol, "d1return":d1return, "d2return":d2return, "%volume":volumeChange, "label":label}, index = [i])
        databysymbol = pd.concat([databysymbol, temp])
    dateonly = []
    for tweetTime in twitterdata["created_at"]:
        tweetTimeElement = datetime.datetime.strptime(tweetTime, '%Y-%m-%d %H:%M:%S')
        dateonly.append(tweetTimeElement.strftime('%Y-%m-%d'))
    dateonly_series = pd.Series(dateonly)
    twitterdata["date"] = dateonly_series.values
    scoreList = []
    for symbol in databysymbol["ticker"]:
        scoreList.append(twitterdata.loc[(twitterdata["date"] == "2017-11-28") & (twitterdata["ticker"] == symbol), "score"].sum())
    scoreList_series = pd.Series(scoreList)
    databysymbol["score"] = scoreList_series.values
    databysymbol.to_csv("dataset6.csv", index = None)
    return databysymbol

def getDataset7(stockdata, twitterdata):
    d110 = stockdata[stockdata.datetime == "2017-11-29 10:00:00"]
    d111 = stockdata[stockdata.datetime == "2017-11-29 11:00:00"]
    d112 = stockdata[stockdata.datetime == "2017-11-29 12:00:00"]
    d113 = stockdata[stockdata.datetime == "2017-11-29 13:00:00"]
    d114 = stockdata[stockdata.datetime == "2017-11-29 14:00:00"]
    d115 = stockdata[stockdata.datetime == "2017-11-29 15:00:00"]
    d116 = stockdata[stockdata.datetime == "2017-11-29 16:00:00"]
    d210 = stockdata[stockdata.datetime == "2017-11-30 10:00:00"]
    d211 = stockdata[stockdata.datetime == "2017-11-30 11:00:00"]
    d212 = stockdata[stockdata.datetime == "2017-11-30 12:00:00"]
    d213 = stockdata[stockdata.datetime == "2017-11-30 13:00:00"]
    d214 = stockdata[stockdata.datetime == "2017-11-30 14:00:00"]
    d215 = stockdata[stockdata.datetime == "2017-11-30 15:00:00"]
    d216 = stockdata[stockdata.datetime == "2017-11-30 16:00:00"]
    dailystock = d110.append(d111, ignore_index = True)
    dailystock = dailystock.append(d112, ignore_index = True)
    dailystock = dailystock.append(d113, ignore_index = True)
    dailystock = dailystock.append(d114, ignore_index = True)
    dailystock = dailystock.append(d115, ignore_index = True)
    dailystock = dailystock.append(d116, ignore_index = True)
    dailystock = dailystock.append(d210, ignore_index = True)
    dailystock = dailystock.append(d211, ignore_index = True)
    dailystock = dailystock.append(d212, ignore_index = True)
    dailystock = dailystock.append(d213, ignore_index = True)
    dailystock = dailystock.append(d214, ignore_index = True)
    dailystock = dailystock.append(d215, ignore_index = True)
    dailystock = dailystock.append(d216, ignore_index = True)
    dailystock = dailystock.sort_values(["ticker", "datetime"])
    dailystock.to_csv("temp1.csv", index = None)
    dailystock = pd.read_csv("temp1.csv")
    databysymbol = pd.DataFrame()
    for i in range(0,3):
        symbol = dailystock["ticker"][14*i]
        d1return = (dailystock.close[14*i+6] - dailystock.open[14*i])/dailystock.open[14*i]
        d2return = (dailystock.close[14*i+13] - dailystock.open[14*i+7])/dailystock.open[14*i+7]
        if d2return >= 0:
            label = "Positive"
        else:
            label = "Negative"
        d1volume = sum(dailystock.volume[14*i:14*i+7])
        d2volume = sum(dailystock.volume[14*i+7:14*i+14])
        volumeChange = d2volume/d1volume - 1
        temp = pd.DataFrame({"ticker":symbol, "d1return":d1return, "d2return":d2return, "%volume":volumeChange, "label":label}, index = [i])
        databysymbol = pd.concat([databysymbol, temp])
    dateonly = []
    for tweetTime in twitterdata["created_at"]:
        tweetTimeElement = datetime.datetime.strptime(tweetTime, '%Y-%m-%d %H:%M:%S')
        dateonly.append(tweetTimeElement.strftime('%Y-%m-%d'))
    dateonly_series = pd.Series(dateonly)
    twitterdata["date"] = dateonly_series.values
    scoreList = []
    for symbol in databysymbol["ticker"]:
        scoreList.append(twitterdata.loc[(twitterdata["date"] == "2017-11-29") & (twitterdata["ticker"] == symbol), "score"].sum())
    scoreList_series = pd.Series(scoreList)
    databysymbol["score"] = scoreList_series.values
    databysymbol.to_csv("dataset7.csv", index = None)
    return databysymbol

def hypotest3(df3, testdf):
    testdata3 = df3[["%volume", "d1return", "score", "label"]]
    validatedf = testdf[["%volume", "d1return", "score", "label"]]
    valueArray1 = testdata3.values
    valueArray2 = validatedf.values
    X_train = valueArray1[:,0:3]
    Y_train = valueArray1[:,3]
    X_validate = valueArray2[:,0:3]
    Y_validate = valueArray2[:,3]
    test_size = 0.25
    seed = 1234
    num_folds = 10
    num_instances = len(X_train)
    seed = 1234
    scoring = "accuracy"
    models = []
    models.append(('NEIGH', KNeighborsClassifier()))
    models.append(('DTCLF', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    models.append(('NB', GaussianNB()))
    models.append(('RFCLF', RandomForestClassifier()))
    results = []
    names = []
    for name, model in models:
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    print()
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, Y_train)
    neighPredictions = neigh.predict(X_validate)
    print("NEIGH", accuracy_score(Y_validate, neighPredictions))
    print(classification_report(Y_validate, neighPredictions))
    dtclf = DecisionTreeClassifier()
    dtclf.fit(X_train, Y_train)
    dtclfPredictions = dtclf.predict(X_validate)
    print("DTCLF", accuracy_score(Y_validate, dtclfPredictions))
    print(classification_report(Y_validate, dtclfPredictions))
    svm = SVC()
    svm.fit(X_train, Y_train)
    svmPredictions = svm.predict(X_validate)
    print("SVM", accuracy_score(Y_validate, svmPredictions))
    print(classification_report(Y_validate, svmPredictions))
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    nbPredictions = nb.predict(X_validate)
    print("NB", accuracy_score(Y_validate, nbPredictions))
    print(classification_report(Y_validate, nbPredictions))
    rfclf = RandomForestClassifier()
    rfclf.fit(X_train, Y_train)
    rfclfPredictions = rfclf.predict(X_validate)
    print("RFCLF", accuracy_score(Y_validate, rfclfPredictions))
    print(classification_report(Y_validate, rfclfPredictions))
    
def main():
    twitterdata = pd.read_csv("final_twitter_data.csv", encoding = 'latin1')
    stockdata = pd.read_csv("final_stock_data.csv")
    dataset4 = getDataset4(stockdata, twitterdata)
    dataset4 = pd.read_csv("dataset4.csv")
    stock = dataset4.loc[3626:3632]
    stock["text"] = "Positive words: " + stock["positive_count"].astype(int).astype(str) + '<br>' +\
                    "Negative words: " + stock["negative_count"].astype(int).astype(str)

    # Plotly daily stock price candlestick with hourly sentiment score
##    trace1 = go.Candlestick(x = stock.datetime,
##                            open = stock.open,
##                            high = stock.high,
##                            low = stock.low,
##                            close = stock.close,
##                            increasing = dict(name = "Increasing"),
##                            decreasing = dict(name = "Decreasing"))
##    trace2 = go.Bar(x = stock.datetime,
##                    y = stock.score,
##                    text = stock.text,
##                    name = "Sentiment Score")
##    fig = tools.make_subplots(rows = 2, cols = 1, shared_xaxes = True)
##    fig.append_trace(trace1, 1, 1)
##    fig.append_trace(trace2, 2, 1)
##    fig["layout"].update(title = "Apple, Inc (AAPL) Oct 30, 2017",
##                         width = 700,
##                         height = 450)
##    py.plot(fig, filename = "simple_ohlc")

    dataset3 = pd.read_csv("dataset3.csv")
    # Plotly Scatter plot: Sentiment score vs. stock return
##    trace3 = go.Scatter(x = dataset3.score,
##                        y = dataset3.d2return,
##                        mode = "markers")
##    data = [trace3]
##    layout = go.Layout(title = "Sentiment Score vs. Stock Change",
##                       width = 700,
##                       height = 450,
##                       xaxis = dict(title = "Sentiment Score (# of Positive words - # of Negative words)"),
##                       yaxis = dict(title = "Stock Price Change (from Oct 30th to 31st, 2017)"))
##    fig = go.Figure(data = data, layout = layout)
##    py.plot(fig, filename = "Scatter:Sentiment Score vs. Stock Change")

    # Group stocks based on prediction results.
    n = len(dataset3)
    TPlist = list() # True Positive (Prediction: Positive; Result: Positive)
    TNlist = list() # True Negative (Prediction: Negative; Result: Negative)
    FPlist = list() # False Positive (Prediction: Positive; Result: Negative)
    FNlist = list() # False Negative (Prediction: Negative; Result: Positive)
    for i in range(n):
        if dataset3.d2return[i] > 0 and dataset3.score[i] > 0:
            TPlist.append(dataset3.ticker[i])
        if dataset3.d2return[i] < 0 and dataset3.score[i] < 0:
            TNlist.append(dataset3.ticker[i])
        if dataset3.d2return[i] < 0 and dataset3.score[i] > 0:
            FPlist.append(dataset3.ticker[i])
        if dataset3.d2return[i] > 0 and dataset3.score[i] < 0:
            FNlist.append(dataset3.ticker[i])

    a = 0.9 # Bubble Scale
    labels = ["True Positive", "True Negative", "False Positive", "False Negative"]
    values = [len(TPlist), len(TNlist), len(FPlist), len(FNlist)]

    # Bubble plot, better visualization from scatter plot before
##    trace9 = go.Scatter(x = [1, -1, 1, -1],
##                        y = [1, -1, -1, 1],
##                        text = labels,
##                        mode = "markers",
##                        name = labels,
##                        marker = dict(color = ['rgb(93, 164, 214)', 'rgb(255, 65, 54)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)'],
##                                      size = [len(TPlist)*a, len(TNlist)*a, len(FPlist)*a, len(FNlist)*a]))
##    data = [trace9]
##    layout = go.Layout(title = "Prediction Results",
##                       xaxis = dict(range = [-2,2]),
##                       yaxis = dict(range = [-2,2]),
##                       width = 700,
##                       height = 450)
##    fig = go.Figure(data = data, layout = layout)
##    py.plot(fig, filename = "bubble")

    # Pie chart with detail information of bubble plot
##    trace8 = go.Pie(labels = labels,
##                    values = values,
##                    hoverinfo = "label+percent",
##                    textinfo = "value",
##                    textfont = dict(size = 20))
##    data = [trace8]
##    layout = go.Layout(title = "Prediction Results",
##                       width = 700,
##                       height = 450)
##    fig = go.Figure(data = data, layout = layout)
##    py.plot([trace8], filename = "pie chart")

    # Study features of "user_verified" and "user_followers"
    trueList = TPlist + TNlist
    falseList = FPlist + FNlist
    trueFollowers = list()
    trueProbVerified = list()
    falseFollowers = list()
    falseProbVerified = list()
    for ticker in trueList:
        subdf = twitterdata.loc[twitterdata["ticker"] == ticker]
        trueFollowers.append(len([a for a in subdf["user_followers"] if a > 2000])/len(subdf))
        trueProbVerified.append(subdf["user_verified"].mean())
    for ticker in falseList:
        subdf = twitterdata.loc[twitterdata["ticker"] == ticker]
        falseFollowers.append(len([a for a in subdf["user_followers"] if a > 2000])/len(subdf))
        falseProbVerified.append(subdf["user_verified"].mean())

    # Two-way t test to compare groups
    sample1t = [a for a in trueProbVerified if (a >= 0.06) and (a <= 0.15)]
    sample1f = [a for a in falseProbVerified if (a >= 0.06) and (a <= 0.15)]
    print('Two-way t test for Verified-User Effect on prediction')
    print(ttest_ind(sample1t, sample1f, equal_var = False))
    print('Two-way t test for User-Followers Effect on prediction')
    print(ttest_ind(trueFollowers, falseFollowers, equal_var = False))

    # Boxplot about follower factor
##    trace4 = go.Box(y = trueFollowers, name = "True")
##    trace5 = go.Box(y = falseFollowers, name = "False")
##    data = [trace4, trace5]
##    layout = go.Layout(title = "User_Followers Feature in True & False Groups",
##                       width = 1200,
##                       height = 450,
##                       xaxis = dict(title = "T&F Groups"),
##                       yaxis = dict(title = "Proportion of Users with over 2000 Followers"))
##    fig = go.Figure(data = data, layout = layout)
##    py.plot(fig, filename = "followers histogram")

    # Boxplot about verify factor
##    trace6 = go.Box(y = trueProbVerified, name = "True")
##    trace7 = go.Box(y = falseProbVerified, name = "False")
##    data = [trace6, trace7]
##    layout = go.Layout(title = "Proportions of Verified Users of True & False Groups",
##                       xaxis = dict(title = "T&F Groups"),
##                       yaxis = dict(range = [0,0.15],
##                                    title = "Proportion of Verified Users"),
##                       width = 1200,
##                       height = 450)
##    fig = go.Figure(data = data, layout = layout)
##    py.plot(fig, filename = "verified histogram")

    apple = pd.read_csv("tweets_AAPL.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])
    bestbuy = pd.read_csv("tweets_BBY.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])
    chevron = pd.read_csv("tweets_CVX.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])
    chipotle = pd.read_csv("tweets_CMG.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])
    costco = pd.read_csv("tweets_COST.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])

    boa = pd.read_csv("tweets_BOA.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])
    fedex = pd.read_csv("tweets_FDX.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])
    mk = pd.read_csv("tweets_KORS.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])
    progressive = pd.read_csv("tweets_PGR.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])
    prudential = pd.read_csv("tweets_PRU.csv", encoding = "latin1", header = None, names = ["datetime", "user_id", "user_name", "user_followers", "user_verified", "user_location", "user_description", "text", "created_at", "source"])

    chipotlet = chipotle.loc[chipotle["user_verified"] == True]
    chipotlef = chipotle.loc[chipotle["user_verified"] == False]
    if len(chipotlet)/len(chipotle) < 0.06:
        remove_n = len(chipotle) - len(chipotlet)*15
        drop_indices = np.random.choice(chipotlef.index, remove_n, replace = False)
        chipotlef = chipotlef.drop(drop_indices)
        dfjoin = [chipotlet, chipotlef]
        chipotle = pd.concat(dfjoin)

    costcot = costco.loc[costco["user_verified"] == True]
    costcof = costco.loc[costco["user_verified"] == False]
    if len(costcot)/len(costco) < 0.06:
        remove_n = len(costco) - len(costcot)*15
        drop_indices = np.random.choice(costcof.index, remove_n, replace = False)
        costcof = costcof.drop(drop_indices)
        dfjoin = [costcot, costcof]
        costco = pd.concat(dfjoin)

    prudentialt = prudential.loc[prudential["user_verified"] == True]
    prudentialf = prudential.loc[prudential["user_verified"] == False]
    if len(prudentialt)/len(prudential) < 0.06:
        remove_n = len(prudential) - len(prudentialt)*15
        drop_indices = np.random.choice(prudentialf.index, remove_n, replace = False)
        prudentialf = prudentialf.drop(drop_indices)
        dfjoin = [prudentialt, prudentialf]
        prudential = pd.concat(dfjoin)
    
    apple["ticker"] = "AAPL"
    bestbuy["ticker"] = "BBY"
    chipotle["ticker"] = "CMG"
    costco["ticker"] = "COST"
    fedex["ticker"] = "FDX"
    mk["ticker"] = "KORS"
    progressive["ticker"] = "PGR"
    prudential["ticker"] = "PRU"
    
    APPLE = pd.read_csv("stock_AAPL.csv")
    BESTBUY = pd.read_csv("stock_BBY.csv")
    CHEVRON = pd.read_csv("stock_CVX.csv")
    CHIPOTLE = pd.read_csv("stock_CMG.csv")
    COSTCO = pd.read_csv("stock_COST.csv")

    BOA = pd.read_csv("stock_BOA.csv")
    FEDEX = pd.read_csv("stock_FDX.csv")
    MK = pd.read_csv("stock_KORS.csv")
    PROGRESSIVE = pd.read_csv("stock_PGR.csv")
    PRUDENTIAL = pd.read_csv("stock_PRU.csv")
    
    apple = clean(apple)
    bestbuy = clean(bestbuy)
    chevron = clean(chevron)
    chipotle = clean(chipotle)
    costco = clean(costco)

    boa = clean(boa)
    fedex = clean(fedex)
    mk = clean(mk)
    progressive = clean(progressive)
    prudential = clean(prudential)

    frames_tweet = [chipotle, costco, prudential]
    frames_stock = [CHIPOTLE, COSTCO, PRUDENTIAL]
    df_tweets = pd.concat(frames_tweet)
    df_tweets.to_csv("temp2.csv", index = None)
    df_stocks = pd.concat(frames_stock)
    df_stocks.columns = ["datetime", "open", "high", "low", "close", "volume", "ticker"]
    stop = getStopWords()
    df_tweets_add = addVariables(df_tweets, stop)
    dataset5 = getDataset5(df_stocks, df_tweets_add)
    dataset6 = getDataset6(df_stocks, df_tweets_add)
    dataset7 = getDataset7(df_stocks, df_tweets_add)
    frames_new = [dataset5, dataset6, dataset7]
    new_dataset = pd.concat(frames_new)
    hypotest3(dataset3, new_dataset)
    
    apple["text"].to_csv("apple.txt", header = None, index = None, sep = "\n")
    bestbuy["text"].to_csv("bestbuy.txt", header = None, index = None, sep = "\n")
    chevron["text"].to_csv("chevron.txt", header = None, index = None, sep = "\n")
    chipotle["text"].to_csv("chipotle.txt", header = None, index = None, sep = "\n")
    costco["text"].to_csv("costco.txt", header = None, index = None, sep = "\n")

    boa["text"].to_csv("boa.txt", header = None, index = None, sep = "\n")
    fedex["text"].to_csv("fedex.txt", header = None, index = None, sep = "\n")
    mk["text"].to_csv("mk.txt", header = None, index = None, sep = "\n")
    progressive["text"].to_csv("progressive.txt", header = None, index = None, sep = "\n")
    prudential["text"].to_csv("prudential.txt", header = None, index = None, sep = "\n")
    
main()
