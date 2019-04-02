#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:08:17 2017

@author: aggrace
"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import metrics
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

myData = pd.read_csv('new_final_twitter_data.csv' , sep=',', encoding='latin1')
newData=pd.concat([myData['userFollowersNumber'], myData['character_length'], 
                      myData['positive_count'], myData['negative_count'], 
                       myData['total_count'], myData['score']], axis=1, 
                      keys=['userFollowersNumber', 'character_length', 'positive_count',
                            'negative_count', 'total_count', 'score'])
x = newData.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalizedDataFrame = pd.DataFrame(x_scaled)
dbscan = DBSCAN(eps=0.03,min_samples=10)
dbscan.fit(normalizedDataFrame)
cluster_labels = dbscan.fit_predict(normalizedDataFrame)
#print(normalizedDataFrame.head(10))
#print(cluster_labels)
# Determine if the clustering is good
silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
print("Using dbscan, the average silhouette_score is :", silhouette_avg)
#PCA deconposition: transfer into 2 dimensional space
pca2D = decomposition.PCA(2)
plot_columns = pca2D.fit_transform(normalizedDataFrame)   
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
plt.show()

## ############################################################################
## Compute DBSCAN
#db = DBSCAN(eps=0.4, min_samples=10).fit(x)
#core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
#core_samples_mask[dbscan.core_sample_indices_] = True
##labels = db.labels_
##
### Number of clusters in labels, ignoring noise if present.
##n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
##
##print('Estimated number of clusters: %d' % n_clusters_)
##print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
##print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
##print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
##print("Adjusted Rand Index: %0.3f"
##      % metrics.adjusted_rand_score(labels_true, labels))
##print("Adjusted Mutual Information: %0.3f"
##      % metrics.adjusted_mutual_info_score(labels_true, labels))
##print("Silhouette Coefficient: %0.3f"
##      % metrics.silhouette_score(x, labels))
#
## Black removed and is used for noise instead.
#unique_labels = set(cluster_labels)
#colors = [plt.cm.Spectral(each)
#          for each in np.linspace(0, 1, len(unique_labels))]
#for k, col in zip(unique_labels, colors):
#    if k == -1:
#        # Black used for noise.
#        col = [0, 0, 0, 1]
#
#    class_member_mask = (cluster_labels == k)
#
#    xy = x[class_member_mask & core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=14)
#
#    xy = x[class_member_mask & ~core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=6)
#
##plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()
