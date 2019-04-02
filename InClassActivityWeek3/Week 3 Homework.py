import sys
from numpy import *
from pandas import *
from pprint import pprint

#def main():
def main(argv):
#    file = open("data.csv", "r")
#    file.from_csv
#    
#    # remove new line symbol
#    line = file.readline()
#    while (line):
#        if line.endswith('\n'):
#            line = line[:-1]
#        data.append(line)
#        line = file.readline()
    
    df = read_csv("data.csv")
#    dfFirst10Lines = df[0:10]
#    dfFirst10Lines.to_csv("OUTPUT_W3.txt", index = False)
#    
#    df5to10 = df[4:10]
#    df5to10col1and3 = df5to10[["Latitude", "Longitude"]].copy()
#    df5to10col1and3.to_csv("OUTPUT_W3.txt", index = False, mode = 'a')
#    
#    dfMDlat = df.Latitude[df["City, State"] == "Annapolis, MD"]
#    dfMDlng = df.Longitude[df["City, State"] == "Annapolis, MD"]
#
#    dfMDlat.to_csv("OUTPUT_W3.txt", index = False, mode = 'a')
#    dfMDlng.to_csv("OUTPUT_W3.txt", index = False, mode = 'a')
    
    error = 0
    for lat in df["Latitude"]:
        if isnull(lat):
            error = error + 1
        try:
            lat = float(lat)
            if lat > 90 or lat < 0:
                error = error + 1
        except ValueError:
            error = error + 1
    print("There are", error, 'unique values for "Latitude" column.')
    
    error = 0
    count = 0
    for lng in df["Longitude"]:
        if isnull(lng):
            error = error + 1
            count = count + 1
        try:
            lng = float(lng)
            if lng > 180 or lng < -180:
                error = error + 1
        except ValueError:
            error = error + 1
            count = count + 1
    print("There are", error, 'unique values for "Longitude" column.')
    print("The number of missing longitudes is", count)
    
    error = 0
    for loc in df["City, State"]:
        if loc[:2].isupper():
            error = error + 1
        if not loc[-2:].isupper():
            error = error + 1
        if loc.count(',') != 1:
            error = error + 1
    print("There are", error, 'unique values for "City, State" column.')
    
    error = 0
    count = 0
    for rate in df["User Ratings"]:
        try:
            rate = int(rate)
        except ValueError:
            error = error + 1
        if rate == 5:
            count = count + 1
    print("There are", error, 'unique values for "User Ratings" column.')
    print("The number of rows that have a user rating of 5 is", count)
    
    df["City"] = ""
    df["State"] = ""
    for i in range(len(df)):
        separate = df["City, State"][i].rfind(',')
        df["City"][i] = df["City, State"][i][:separate]
        df["State"][i] = df["City, State"][i][separate+2:]
        
    df["State"] = df["State"].str.lower()
    df["City"] = df["City"].str.lower()
    
    count = 0
    c = 0
    for i in range(len(df)):
        if isnull(df["Latitude"][i]) or isnull(df["Longitude"][i]):
            print("Row dropped is in", df["City, State"][i])
            df = df.drop(i)
            count = count + 1
        else:
            try:
                float(df["Latitude"][i])
                float(df["Longitude"][i])
                int(df["User Ratings"][i])
            except ValueError:
                print("Row dropped is in", df["City, State"][i])
                df = df.drop(i)
                count = count + 1
    print(count, "rows are dropped because of either missing observations or invalid user rating.")
    
    df = df.sort_values(by=["Latitude", "Longitude"])
    
    print(df[0:10])
    
    df.to_csv("netid.txt", index = False, sep = '|')
if __name__=="__main__":
    main(sys.argv)
#main()
