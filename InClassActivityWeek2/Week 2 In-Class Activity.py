from requests import *
from pandas import *

def main():
    with open('INPUT.txt') as file:
        zipcode = file.read().splitlines()
        
    BaseURL = "http://www.airnowapi.org/aq/observation/zipCode/historical/"
    for z in zipcode:
        URLPost = {'API_KEY' : '********************************',  # API key hide on purpose
                   'format' : 'application/json',
                   'zipCode' : z,
                   'date' : '2017-09-11T00-0000',
                   'distance' : '25'}
        response = get(BaseURL, URLPost)
        jsontxt = response.json()
        jsondf = DataFrame(jsontxt)
        jsondf["ZipCode"] = z
        df = jsondf[['ZipCode', 'DateObserved', 'StateCode', 'ReportingArea', 'ParameterName', 'AQI']].copy()
        if z == zipcode[0]:
            df.to_csv("AQI_output.csv", header = True, index = False, sep = ',')
        else:
            df.to_csv("AQI_output.csv", header = False, index = False, sep = ',', mode = 'a')

    with open('INPUT2.txt') as file2:
        latlngList = file2.read().splitlines()
        
    BaseURL2 = "http://www.airnowapi.org/aq/observation/latLong/historical/"
    for latlng in latlngList:
        ll = latlng.split(',')
        URLPost2 = {'API_KEY' : '********************', # API key hide on purpose
                   'format' : 'application/json',
                   'latitude' : ll[0],
                   'longitude' : ll[1],
                   'date' : '2017-09-11T00-0000',
                   'distance' : '76'}
        response2 = get(BaseURL2, URLPost2)
        jsontxt2 = response2.json()
        jsondf2 = DataFrame(jsontxt2)
        df2 = jsondf2[['Latitude', 'Longitude', 'DateObserved', 'StateCode', 'ReportingArea', 'ParameterName', 'AQI']].copy()
        if latlng == latlngList[0]:
            df2.to_csv("AQI_output2.csv", header = True, index = False, sep = ',')
        else:
            df2.to_csv("AQI_output2.csv", header = False, index = False, sep = ',', mode = 'a')
            
main()
