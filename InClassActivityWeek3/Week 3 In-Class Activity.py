from numpy import *
from pandas import *
from pprint import pprint

def main():
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
    dfFirst10Lines = df[0:10]
    dfFirst10Lines.to_csv("OUTPUT_W3.txt", index = False)
    
    df5to10 = df[4:10]
    df5to10col1and3 = df5to10[["Latitude", "Longitude"]].copy()
    df5to10col1and3.to_csv("OUTPUT_W3.txt", index = False, mode = 'a')
    
    dfMDlat = df.Latitude[df["City, State"] == "Annapolis, MD"]
    dfMDlng = df.Longitude[df["City, State"] == "Annapolis, MD"]

    dfMDlat.to_csv("OUTPUT_W3.txt", index = False, mode = 'a')
    dfMDlng.to_csv("OUTPUT_W3.txt", index = False, mode = 'a')
    
main()
