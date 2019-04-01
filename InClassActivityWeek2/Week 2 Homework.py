from pandas import *
from requests import *

def main():
    with open("BBCURLwithapiKey.txt") as file:  # api key in file hidden on purpose
        URL = file.readline()
    
    response = get(URL)             # text file is a single line of a complete URL
    jsontxt = response.json()       # jsontxt: read API as text version
    jsondict = jsontxt["articles"]  # jsondict: remove outer dictionary
    jsondf = DataFrame(jsondict)    # jsondf: construct json dataframe
    df = jsondf[["publishedAt", "author", "title", "description"]].copy()
    df.index = range(1, len(df) + 1)
    for i in range(len(df)):
        df1row = df.loc[str(i+1):str(i+1), "publishedAt":"title"]
        df2row = df.loc[str(i+1):str(i+1), "description":"description"]
        df1row.to_csv("BBCtopOutput.txt", header = False, index = True, sep = '|', mode = 'a')
        df2row.to_csv("BBCtopOutput.txt", header = False, index = True, sep = '|', mode = 'a')
        
main()
