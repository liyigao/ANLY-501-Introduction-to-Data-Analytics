#! /bin/usr/python
"""
@author: Yigao Li
@author: Qixu Cao
@author: Taoran Yu
@author: Xiaoman Dong
"""

import re
import pandas as pd
import json
import numpy as np
import datetime
import pytz

# Objective Naive Score
def get_r_clean_naive(df):
    num_column_df = len(df.columns)
    #print(num_column_df)
    num_row_df = len(df.index)
    #print(num_row_df)
    # Get the isnull() count for each columns in a dictionary
    null_df_info = df.isnull().sum()
    #print(null_myData_info)
    r_clean_column_sum = 0
    for column_label, count_null_per_column in null_df_info.items():
        # Calculate the naive clean rate for each column
        r_clean_column = (num_row_df - count_null_per_column) / num_row_df
        print("r_clean for " + str(column_label) + " is: " + str(r_clean_column))
        r_clean_column_sum = r_clean_column_sum + r_clean_column
    # Calculate the average
    r_clean_naive = r_clean_column_sum / num_column_df
    #print(r_clean_naive)
    return r_clean_naive

# Subjective Logical Score
# Data Clean
# Remove rows with all NaN
def cleanNaN(dataframe):
    dataframe = dataframe.dropna(axis=0, how='all')
    return dataframe

# Data Process
# Formatting time in Twitter dataset to meet with Stock price dataset. No additional formatting needed for Stock Price dataset
def cleanTimeFormat(dt_column):
    newtime = []
    for t in dt_column:
        t_element = datetime.datetime.strptime(t, '%a %b %d %H:%M:%S %z %Y')
        t_zone = t_element.astimezone(pytz.timezone('US/Eastern'))
        newtime.append(t_zone.strftime('%Y-%m-%d %H:%M:%S')) #'%m/%d/%Y %H:%M:%S'
    return newtime

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

if __name__ == "__main__":
    # READ DATA
    #csvfile_twitter = 'S&P500_twitter_test.csv'
    #csvfile_twitter = 'Workbook2.csv'
    csvfile_twitter = 'S_P500_twitter.csv'
    csvfile_stock = 'S&P500_stock_price.csv'
    data_twitter = pd.read_csv(csvfile_twitter, sep=',', encoding='latin1')
    data_stock = pd.read_csv(csvfile_stock, sep=',', encoding='latin1')
    #print(myData.iloc[:10])

    # Data Clean Part 1. Objective Naive Score
    print("###################################################")
    print("##               Data Clean Part 1               ##")
    print("##             Objective Naive Score             ##")
    print("###################################################")
    # Calculate the Objective Naive Score
    row_number_data_twitter_raw = len(data_twitter.index)
    print("\nTotal rows in raw twitter data:", row_number_data_twitter_raw)
    print("\nObjective Naive Score for raw twitter data:")
    print(get_r_clean_naive(data_twitter))
    row_number_data_stock_raw = len(data_stock.index)
    print("\nTotal rows in raw stock data:",row_number_data_stock_raw)
    print("\nObjective Naive Score for raw stock data:")
    print(get_r_clean_naive(data_stock))

    # Clean NaN data
    twitter_df1 = data_twitter
    stock_df1 = data_stock
    print("\nCleanning NaN data ...")
    #twitter_df1 = cleanNaN(twitter_df1)
    # Since this project needs twitter text, it's critical to make sure the r_clean_naive of twitter['text']=1
    twitter_df1 = twitter_df1.dropna(subset=['text'])
    stock_df1 = cleanNaN(stock_df1)
    row_number_data_twitter_df1 = len(twitter_df1.index)
    print("\nTotal rows of cleaned twitter data:",row_number_data_twitter_df1)
    print("# of rows removed:", (row_number_data_twitter_raw - row_number_data_twitter_df1))
    print("\nObjective Naive Score for cleaned twitter data:")
    print(get_r_clean_naive(twitter_df1))
    #twitter_df1.to_csv("Cleaned_twitter_data.csv")
    row_number_data_stock_df1 = len(stock_df1.index)
    print("\nTotal rows of cleaned stock data:",row_number_data_stock_df1)
    print("# of rows removed:", (row_number_data_stock_raw - row_number_data_stock_df1))
    print("\nObjective Naive Score for cleaned stock data:")
    print(get_r_clean_naive(stock_df1))

    # Data Clean Part 2. Subjective Logical Score
    print("###################################################")
    print("##               Data Clean Part 2               ##")
    print("##             Subjective Logical Score          ##")
    print("###################################################")

    twitter_df2 = twitter_df1
    stock_df2 = stock_df1

    print("\nRemoving characters other than letters, numbers and punctuations (!,?,#) in twitter['text']")
    print("Replacing those characters with ' ' to calculate the r_clean_logical....")
    # Remove strings other than letters, numbers, #, ?, !
    twitter_df2['text'] = twitter_df2['text'].apply(process_String_Keep_Only_Letters_Numbers)
    twitter_df2['user_description'] = twitter_df2['user_description'].apply(process_String_Keep_Only_Letters_Numbers)
    twitter_df2['user_location'] = twitter_df2['user_location'].apply(process_String_Keep_Only_Letters_Numbers)
    # Remove twitter text length less than 5 characters
    twitter_df2['text'] = twitter_df2['text'].apply(process_String_Length_Less_Than_5)
    # Replace empty string to NaN
    twitter_df2['text'] = twitter_df2['text'].apply(lambda x: x.strip()).replace('', np.nan)
    #twitter_df2 = twitter_df2.apply(lambda x: x.replace('', np.nan))
    #twitter_df2 = twitter_df2.replace(r'^\s+$', np.nan, regex=True)
    #print(twitter_df2.iloc[770:780])
    print(twitter_df2[pd.isnull(twitter_df2['text'])])
    print("\nAfter removing unuseful tweets, here is the Subjective Logical Score: ")
    row_number_data_twitter_df2 = len(twitter_df2.index)
    print("\nTotal rows in current twitter data:", row_number_data_twitter_df2)
    print("\nSubjective Logical Score for current twitter data:")
    print(get_r_clean_naive(twitter_df2))

    # Clean NaN data generated by unuseful tweets
    print("\nCleanning NaN data ...")
    # Since this project needs twitter text, it's critical to make sure the r_clean_naive of twitter['text']=1
    twitter_df2 = twitter_df2.dropna(subset=['text'])
    row_number_data_twitter_df2 = len(twitter_df2.index)
    print("\nTotal rows of cleaned twitter data:",row_number_data_twitter_df2)
    print("# of rows removed:", (row_number_data_twitter_df1 - row_number_data_twitter_df2))
    print("\nObjective Naive Score for cleaned twitter data:")
    print(get_r_clean_naive(twitter_df2))

    # Data Clean Part 3. Transform the data
    print("###################################################")
    print("##               Data Clean Part 3               ##")
    print("##              Transform the data               ##")
    print("###################################################")

    twitter_df3 = twitter_df2
    twitter_df3 = twitter_df3.dropna(subset=['created_at'])
    stock_df3 = stock_df2

    ten_row_data_twitter = twitter_df3.iloc[:10]
    ten_row_data_stock = stock_df3.iloc[:10]
    #print(ten_row_data_twitter)

    # Unify the time format between twitter and stock data
    print("\nChanging time format for twitter['created_at']....")
    twitter_df3['created_at'] = cleanTimeFormat(twitter_df3['created_at'])
    print("Done!")
    # Convert all letters in twitter['text'] to small case
    print("\nConverting all letters in twitter['text'] to small case...")
    twitter_df3['text'] = twitter_df3['text'].apply(process_String_To_Small_Letters)
    print("Done!")
    print("\nGenerating new csv file for clean data...")
    twitter_df3.to_csv("clean_twitter_data.csv")
    stock_df3.to_csv("clean_stock_data.csv")
    #print(twitter_df3.iloc[:3])
    #print(stock_df3.iloc[:3])
