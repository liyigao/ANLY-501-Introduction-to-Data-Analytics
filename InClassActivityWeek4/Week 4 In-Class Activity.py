import sys
from pandas import *

def main(argv):
    
    df = read_csv("NY_Times_LARGE.csv")
    print(df[0:10])
    
if __name__=="__main__":
    main(sys.argv)
