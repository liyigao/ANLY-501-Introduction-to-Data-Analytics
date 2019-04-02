# In Class Activity Week 4 - Variable Creation and Descriptive Analysis

## In-class Portion

**Download the dataset**.  The dataset is one day’s data collection of clicks recorded for the New York Times homepage in 2012. Each row is a single user. In other words, a user entered the New York Times home page and clicked various links on that page. The dataset is a record of specific activity per user on one day. 

The **“Impressions”** are the number of advertisements displayed (shown) to the user while the user was on the web page.  
The **“Clicks”** are the number of advertisements that the user actually clicked on.   
Therefore, the number of Clicks must be less than or equal to the number of  Impressions.   (Clicks <= Impressions)

### Instructions

1.	Write Python3 code to read in the dataset (into a pandas dataframe). Remember to read in a reduced size sample of the dataset first (say the first 100 rows). 

2.	Print to the screen the first 10 rows of the data. 

## Homework Portion

1. Plot the distribution (histogram) of the number of impressions for gender. How does this compare to distribution impressions by age categories (histogram for each age group)? Be sure to write code to create a histogram for impressions and gender as well as a set of histograms for impressions and age group.

2. Write code to try at least three (3) different values for k for the k-means clustering on the data. What do you see?

3. Use the Silhouette procedure (also in scikit-learn) to measure the cluster quality for the 3 clusterings you created (for each value of k). What do you find?

4. Add the click behavior data to the k-means analysis (you will need to create a numeric variable for it). Once you add the click behavior, repeat numbers 2 and 3 above. How does that alter the result?
