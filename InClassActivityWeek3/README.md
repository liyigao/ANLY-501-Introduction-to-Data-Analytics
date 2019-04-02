# In Class Activity Week 3 - Data Wrangling: Munging and Cleaning

## In-class Portion

1. Download the data file.

2. Write Python3 code to read in the datafile into a pandas dataframe.

3. Write the following to an output file and call the output file OUTPUT_W3.txt

a. Write the first 10 lines (rows) of the data to the output file.

b. Write rows 5 - 10 only for columns 1 and 3 to the output file.

c. Write the latitude and longitude for Annapolis, MD.

## Homework Portion

### Step 1 - Analyzing the Data

Write the Python3 code and print out the answers to the screen and (if noted) to an output file called netid.txt. Do this for the following:

a.	After you load in the data, determine the number of unique values for each column. Print this information to the screen (do not write this to your file).

b.	Find the number of rows that have a user rating of 5. Print this information to the screen (do not write this to your file). 

c.	Find the number of missing longitudes. Print this information to the screen (do not write this to your file). 

### Step 2 - Cleaning the Data

Write a  final “cleaned” dataframe to your output file called netid.txt.

**The delimiter should be a bar (|), rather than a comma or space. Do all of the following:**

a.      Split the city and the state into two separate columns.

b.      Update all the cities and states to make them lower case.

c.      Drop rows with missing observations. Print to the screen the number of rows you dropped and the City and State for that row (if available). 

d.      Remove rows with invalid user rating. Print to the screen the number of rows you dropped and the City and State for that row (if available). 

e.      Sort data by latitude and longitude. Print to the screen the first 10 sorted results.
