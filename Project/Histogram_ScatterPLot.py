#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 06:08:25 2017

@author: aggrace
"""

import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

dataset=pd.read_csv('final_stock_data.csv')
dataset1=pd.read_csv('new_final_twitter_data.csv',encoding='latin1')
mydata=dataset1[['user_followers', 'user_verified','total_count']]

#Histogram of followers of verified user
subset = dataset['volumeRange']
subset.hist()
plt.figure(1)
titleLabel="Distribution of stock volume"
plt.title(titleLabel)
plt.xlabel("Stock volumn")
plt.ylabel("number of stock indexes")
plt.show()

#Histogram of distribution of non-verified/verified users
subset1=dataset1['user_verified']
subset1.hist()
plt.figure(2)
titleLabel1="Distribution of non-verified/verified users"
plt.title(titleLabel1)
plt.xlabel("non-verified/verified users")
plt.ylabel("number of users")
plt.show()

#Histogram of distribution of stock price range
subset2=dataset['priceRange']
subset2.hist()
plt.figure(3)
titleLabel2="Distribution of stock price range"
plt.title(titleLabel2)
plt.xlabel("priceRange")
plt.ylabel("number of stocks")
plt.show()

#Histogram of distribution of twitter followers volume
subset3=dataset1['userFollowersNumber']
subset3.hist()
plt.figure(4)
titleLabel3="Distribution of followers volume"
plt.title(titleLabel3)
plt.xlabel("followersVolume")
plt.ylabel("number of followers")
plt.show()

#Find the correlation among user followers, user_verified and total words count
print(mydata.corr())

#plot the scatterplot for user_verified and user_followers
plt.figure(5)
titleLabel="Scatterplot of verified/non verified user and user followers"
plt.title(titleLabel)
plt.scatter(mydata['user_verified'],mydata['user_followers'])
plt.xlabel("non verified user/verified user")
plt.ylabel("user followers")

#plot the scatterplot for user followers and total_count
plt.figure(6)
titleLabel="Scatterplot of followers and total count words"
plt.title(titleLabel)
plt.scatter(mydata['user_followers'],mydata['total_count'])
plt.xlabel("number of followers")
plt.ylabel("total words count")

#plot the scatterplot for user_verified with total_count
plt.figure(7)
titleLabel="Scatterplot of verified user/non verified user and total words count"
plt.title(titleLabel)
plt.scatter(mydata['user_verified'],mydata['total_count'])
plt.xlabel("non verified user/verified user")
#plt.ylabel("total words count")

#scatterplot in 3d
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

x=mydata['user_verified']
y=mydata['user_followers']
z=mydata['total_count']

ax.scatter(x,y,z,c='r',marker='o')

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

plt.figure(8)
titleLabel="3D Scatterplot"
plt.title(titleLabel)
plt.xlabel("non verified user/verified user")
plt.ylabel("number of followers")
plt.show()

#Show the correlation by visualization
plt.matshow(mydata.corr())
