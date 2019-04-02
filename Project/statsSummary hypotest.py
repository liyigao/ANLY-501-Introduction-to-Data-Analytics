import pandas as pd
import nltk
import datetime
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

## Dealing with data
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

## Merge 2 datasets
# Dataset1
# Merge twitter data into stock data
# Group tweets for a single stock in 1 hours.
# Variables: datatime, open, high, low, close, volume, ticker, priceRange, volumeRange, positive_count, negative_count, total_count, score
def getDataset1(stockdata, twitterdata):
    time_modified = []
    for tweetTime in twitterdata["created_at"]:
        tweetTimeElement = datetime.datetime.strptime(tweetTime, '%Y-%m-%d %H:%M:%S')
        tweetTimeElement = tweetTimeElement.replace(minute = 0, second = 0)
        tweetTimeElement = tweetTimeElement + datetime.timedelta(hours = 1)
        time_modified.append(tweetTimeElement.strftime('%Y-%m-%d %H:%M:%S'))
    time_modified_series = pd.Series(time_modified)
    twitterdata["datetime"] = time_modified_series.values
    tdf1 = twitterdata[["ticker", "positive_count", "datetime"]]
    tdf1 = tdf1.groupby(["ticker", "datetime"]).sum()
    tdf2 = twitterdata[["ticker", "negative_count", "datetime"]]
    tdf2 = tdf2.groupby(["ticker", "datetime"]).sum()
    tdf = pd.concat([tdf1, tdf2], axis = 1).reset_index()
    tdf["total_count"] = tdf["positive_count"] + tdf["negative_count"]
    tdf["score"] = tdf["positive_count"] - tdf["negative_count"]
    dataset1 = stockdata.merge(tdf, on = ["ticker", "datetime"])
    dataset1.to_csv("dataset1.csv", index = None)
    print(dataset1.head())
    return dataset1

# Dataset2
# Merge twitter and stock data into new dataframe of rows of stock symbols
# Variables: %volume, d1return, d2return, label(Positive/Negative), ticker, score
# Day1=10/30/2017, Day2=10/31/2017
def getDataset3(stockdata, twitterdata, stopSymbol = "TPR"):
    stockdata = stockdata[stockdata.ticker != stopSymbol]
    d110 = stockdata[stockdata.datetime == "2017-10-30 10:00:00"]
    d111 = stockdata[stockdata.datetime == "2017-10-30 11:00:00"]
    d112 = stockdata[stockdata.datetime == "2017-10-30 12:00:00"]
    d113 = stockdata[stockdata.datetime == "2017-10-30 13:00:00"]
    d114 = stockdata[stockdata.datetime == "2017-10-30 14:00:00"]
    d115 = stockdata[stockdata.datetime == "2017-10-30 15:00:00"]
    d116 = stockdata[stockdata.datetime == "2017-10-30 16:00:00"]
    d210 = stockdata[stockdata.datetime == "2017-10-31 10:00:00"]
    d211 = stockdata[stockdata.datetime == "2017-10-31 11:00:00"]
    d212 = stockdata[stockdata.datetime == "2017-10-31 12:00:00"]
    d213 = stockdata[stockdata.datetime == "2017-10-31 13:00:00"]
    d214 = stockdata[stockdata.datetime == "2017-10-31 14:00:00"]
    d215 = stockdata[stockdata.datetime == "2017-10-31 15:00:00"]
    d216 = stockdata[stockdata.datetime == "2017-10-31 16:00:00"]
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
    for i in range(0,504):
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
        scoreList.append(twitterdata.loc[(twitterdata["date"] == "2017-10-30") & (twitterdata["ticker"] == symbol), "score"].sum())
    scoreList_series = pd.Series(scoreList)
    databysymbol["score"] = scoreList_series.values
    databysymbol.to_csv("dataset3.csv")
    print(databysymbol.head())
    return databysymbol
    
### Basic Statistical Analysis and data cleaning insight (15%)

# Determine mean, mode, median, standard deviation
def statsSummary(df):
    return df.describe()

### Hypothesis Testing (30%)

## Hypothesis Testing 1 (ANOVA test)
# Sentiment expressions on Twitter vs. Stock trading volume within the same period of time
# Null Hypothesis: Mean of total sentiment word count per hr is the same for stocks in each volume group
# Alternative Hypothesis: Mean value is not the same for each volume group
def hypotest1(df1):
    # 1st time: include data with 0 sentiment word count
    testdata1 = df1[["volumeRange", "total_count"]]
    groups1 = testdata1.groupby("volumeRange")["total_count"].apply(list)
    LargeList1 = groups1.tolist()
    volume1_1e4 = LargeList1[0]
    volume1_1e5 = LargeList1[1]
    volume1_1e6 = LargeList1[2]
    volume1_1e7 = LargeList1[3]
    volume1_1e8 = LargeList1[4]
    volume1_1e9 = LargeList1[5]
    anova1 = stats.f_oneway(volume1_1e4, volume1_1e5, volume1_1e6, volume1_1e7, volume1_1e8, volume1_1e9)
    print("ANOVA result for data w/ total_count=0")
    print(anova1)
    # 2nd time: exclude data with 0 sentiment word count
    testdata1 = testdata1[testdata1.total_count != 0]
    groups2 = testdata1.groupby("volumeRange")["total_count"].apply(list)
    LargeList2 = groups2.tolist()
    volume2_1e4 = LargeList2[0]
    volume2_1e5 = LargeList2[1]
    volume2_1e6 = LargeList2[2]
    volume2_1e7 = LargeList2[3]
    volume2_1e8 = LargeList2[4]
    volume2_1e9 = LargeList2[5]
    anova2 = stats.f_oneway(volume2_1e4, volume2_1e5, volume2_1e6, volume2_1e7, volume2_1e8, volume2_1e9)
    print("ANOVA result for data w/o total_count=0")
    print(anova2)

## Hypothesis Testing 2 (Linear Regression)
# Total sentiment word count is a linear combination of tweet character length and verified user(Yes:1, No:0)
# total_count = beta0 + beta1 * user_verified + beta2 * character_length
# Null Hypothesis: beta1 = beta2 = 0
# Alternative Hypothesis: betaj != 0 for either j
def hypotest2(twitterdata):
    df2 = twitterdata
    df2_lm = smf.ols("total_count ~ character_length + C(user_verified)", data = df2).fit()
    print(df2_lm.summary())
    print(sm.stats.anova_lm(df2_lm, typ = 2))

## Hypothesis Testing 3 (Supervised Learning)
# Class: Stock return in the future (next day)
# Variables: Percentage Change in volume in 2 days (today vs. next day)
#            Stock return from today
#            Sentiment score (Positive word count subtract(-) Negative)
# Machine Learning methods: K Nearest Neighbors
#                           Decision Tree
#                           Support Vector Machine
#                           Naive Bayes
#                           Random Forest
def hypotest3(df3):
    testdata3 = df3[["%volume", "d1return", "score", "label"]]
    valueArray = testdata3.values
    X = valueArray[:,0:3]
    Y = valueArray[:,3]
    test_size = 0.25
    seed = 1234
    X_train, X_validate, Y_train, Y_validate = cross_validation.train_test_split(X, Y, test_size = test_size, random_state = seed)
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
    stop = getStopWords()
    stockdata = pd.read_csv("clean_stock_data.csv")
    twitterdata = pd.read_csv("clean_twitter_data.csv", encoding='latin1')
    twitterdata_new = addVariables(twitterdata, stop)
    twitterdata_new = pd.read_csv("complete_twitter_data.csv", encoding = "latin1")
    stockdf = stockdata[["1. open", "2. high", "3. low", "4. close", "5. volume"]]
    twitterdf = twitterdata_new[["user_followers", "character_length", "positive_count", "negative_count", "total_count", "score"]]
    print("Summary Statistics in stock data\n")
    print(statsSummary(stockdf))
    print("\n")
    print("Summary Statistics in twitter data\n")
    print(statsSummary(twitterdf))
    stockdata = pd.read_csv("final_stock_data.csv")
    twitterdata = pd.read_csv("final_twitter_data.csv", encoding = 'latin1')
    stockdata = stockdata[["Unnamed: 0.1.1", "1. open", "2. high", "3. low", "4. close", "5. volume", "symbol", "priceRange", "volumeRange"]]
    stockdata.columns = ["datetime", "open", "high", "low", "close", "volume", "ticker", "priceRange", "volumeRange"]
    counts = stockdata["ticker"].value_counts()
    print(counts)
    # Remove TPR
    dataset1 = getDataset1(stockdata, twitterdata)
    dataset3 = getDataset3(stockdata, twitterdata)
    hypotest1(dataset1)
    hypotest2(twitterdata)
    hypotest3(dataset3)
    
main()
