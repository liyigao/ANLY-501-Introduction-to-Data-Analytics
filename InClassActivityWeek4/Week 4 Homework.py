import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

# basic summary for data
def summary(data):
    
    return data.info()
    return data.describe()

# add age binning column based on given bins
def addAgebin(data, bins):
    
    names = range(1,len(bins))
    data['AgeGroups'] = pd.cut(data['Age'], bins, labels=names)
    data['AgeGroups1'] = np.digitize(data['Age'], bins)
    data['BinRanges'] = pd.cut(data['Age'], bins)
    pprint(data['BinRanges'].value_counts())

def addClickThroughRate(data):
    
    data['ClickThrougRate'] = data['Clicks'] / data['Impressions']
    
def databoxplot(data):
    
    plt.style.use = 'default'
    data.boxplot()
    plt.title("Box plots of all variables in data set")
    plt.show()
    plt.savefig('boxPlot.png')
    
def datahist(data):
    
    data.hist()
    plt.title("Distribution of different variables")
    plt.show()
    plt.savefig('dataHistogram.png')
    
def datahistAge(data):
    
    data['AgeGroups1'].hist()
    plt.title("Distribution of Age")
    plt.show()
    plt.savefig('AgeDistribution.png')
    
def datahistAgeGroups(data):
    
    NewDF=pd.concat([data['AgeGroups1'], data['Impressions']], axis=1, keys=['AgeGroups1', 'Impressions'])
    print(NewDF)
    
def datahistVars(data):
    
    VariableList=["Gender", "Clicks", "Signed_In", "Impressions"]
    for var in VariableList:  
        name="Week4_5ICA_Graph_for_"+var
        #RE: NYdataframe['Impressions'].hist(by=NYdataframe['age_group'])
        data[var].hist(by=data['AgeGroups1'])
        pl.suptitle("Histograms by Age Group for " + var)
        plt.show()
        plt.savefig(name)

# Histograms of the number of impressions for age groups
def datahistImpressionforAgeGroups(data):
    
    ageSeries = data['AgeGroups1'].unique()
    ageSeries.sort()
    counter = 1
    for age in ageSeries:
        pprint(age)
    
        queryString = "AgeGroups1 == " + str(age)
        ageGroupImpressions = data[['AgeGroups1', 'Impressions']].query(queryString)
    
        ageGroupImpressions['Impressions'].hist()
        titleLabel = "Distribution of Impressions for Age Group " + str(age)
        plt.title(titleLabel)
        plt.xlabel("Number of Impressions")
        plt.ylabel("Frequency")
        plt.show()
        
        counter += 1

# Histograms of the number of impressions for genders
def datahistImpressionforGender (data):
    genderSeries = data['Gender'].unique()
    counter = 1
    for gender in genderSeries:
        queryString = "Gender == " + str(gender)
        genderImpressions = data[['Gender', 'Impressions']].query(queryString)
    
        genderImpressions['Impressions'].hist()
        titleLabel = "Distribution of Impressions for Gender " + str(gender)
        plt.title(titleLabel)
        plt.xlabel("Number of Impressions")
        plt.ylabel("Frequency")
        plt.show()
    
# create a new variable that categorizes behavior based on click-thru rate.
def click_behavior (row):
   if row['Clicks'] > 0 :
      return 'Clicks'
   if row['Impressions'] == 0 :
      return 'noImpressions'
   if row['Impressions'] > 0 :
      return 'Impressions'

   return 'Other'

# Create categories for click thru behavior. 
def addClick_behavior (data):
    
    data['clickBehavior']  = myData.apply (lambda row: click_behavior (row),axis=1)
    pprint(myData['clickBehavior'].value_counts())

# fix data - make sure data only have numeric values for kmeans, with ClickThroughRate column
def fixwCTR(data):
    
    data = pd.concat([data['Age'], data['Gender'], data['Impressions'], data['Clicks'], data['Signed_In'], data['AgeGroups1'], data['ClickThrougRate']], 
                 axis=1, keys=['Age', 'Gender', 'Impressions', 'Clicks', 'Signed_In','AgeGroups1', 'ClickThrougRate' ])
    x = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    return normalizedDataFrame

# create clusters, with ClickThroughRate column
def clusterswCTR(k, data):
    
    kmeans = KMeans(n_clusters=k)
    d = fixwCTR(data)
    cluster_labels = kmeans.fit_predict(d)
    centroids = kmeans.cluster_centers_
    print(cluster_labels)
    print(centroids)
    prediction = kmeans.predict(d)
    print(prediction)

# determine if the clustering is good, with ClickThroughRate column
def silhouettewCTR(k, data):
    
    kmeans = KMeans(n_clusters=k)
    d = fixwCTR(data)
    cluster_labels = kmeans.fit_predict(d)
    silhouette_avg = silhouette_score(d, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

# fix data - make sure data only have numeric values for kmeans, without ClickThroughRate column
def fixwoCTR(data):
    
    data = pd.concat([data['Age'], data['Gender'], data['Impressions'], data['Clicks'], data['Signed_In'], data['AgeGroups1']], 
                 axis=1, keys=['Age', 'Gender', 'Impressions', 'Clicks', 'Signed_In','AgeGroups1'])
    x = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    return normalizedDataFrame

# create clusters, without ClickThroughRate column
def clusterswoCTR(k, data):
    
    kmeans = KMeans(n_clusters=k)
    d = fixwoCTR(data)
    cluster_labels = kmeans.fit_predict(d)
    centroids = kmeans.cluster_centers_
    print(cluster_labels)
    print(centroids)
    prediction = kmeans.predict(d)
    print(prediction)

# determine if the clustering is good, with ClickThroughRate column
def silhouettewoCTR(k, data):
    
    kmeans = KMeans(n_clusters=k)
    d = fixwoCTR(data)
    cluster_labels = kmeans.fit_predict(d)
    silhouette_avg = silhouette_score(d, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    
def main(argv):
    
    #df = pd.read_csv("NY_Times_SMALL.csv")
    df = pd.read_csv("NY_Times_LARGE.csv")
    bins = [0, 15, 25, 40, 60, 1000]
    addAgebin(df, bins)
    df = df.dropna()
    datahistImpressionforGender(df)
    datahistImpressionforAgeGroups(df)
    for k in [3,6,9]:
        print('k =', k)
        clusterswoCTR(k, df)
        silhouettewoCTR(k, df)
    addClickThroughRate(df)
    df = df.dropna()
    for k in [3,6,9]:
        print('k =', k)
        clusterswCTR(k, df)
        silhouettewCTR(k, df)
    
if __name__=="__main__":
    main(sys.argv)
