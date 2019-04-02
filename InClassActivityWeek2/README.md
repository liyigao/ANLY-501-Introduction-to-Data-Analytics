# In Class Activity Week 2 - Using APIs

## In-class Portion

**Go to the AirNow API Site**:  https://docs.airnowapi.org/

1. By reading the Site, determine how to get an API “key” which is required for API use

2. Register for the site, log in, and create a KEY.  As with most API Sites, there are data use guidelines and rate limitations – review these.

3. From the Web Services link (https://docs.airnowapi.org/webservices), investigate the query options.

### Instructions

1. Use your API Key to write a Python 3 program to directly query the AirNow API for AQI data.

a. Your program should read in data from an input file that contains any 10 zip codes and should gather AQI data for each zip code location (include at least ozone and PM2.5). Name the input file, INPUT.txt.

b. Your program should print out a table of the results and should create a .csv file of the results. Name file AQI_output.csv.

2. Repeat all of the above using latitude and longitude instead. Create the appropriate input and output files.

## Homework Portion

Learn about the News API and get an API Key: https://newsapi.org/

Write Python code to read data from the request URL into a json file. You can use .txt for this or .json. Then - into a second text file - generate the following from the json you collected: (using the following text format)

`Article number|Date|Author|Title`  
`Article number|Contents of article - description`

**Pretend Examples: (This is a pretend example of what your generated file will look like after you get and process the json results)**

`1|2-22-2017|Ami Gates|Data Science Opinion Poll`  
`1|Random information about the opinion poll...`  
`2|3-12-2017|John Smith|Data is Awesome`  
`2|This article is….`  
`3| …. Etc. etc...`  

