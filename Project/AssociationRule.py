#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:35:28 2017

@author: aggrace
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 00:59:30 2017

@author: aggrace
"""

"import library"
import pandas as pd
import numpy as np

"Load dataset"
dataset = pd.read_csv('new_final_twitter_data.csv',encoding="latin1")

"we only want 'ticker' and 'user_location' column data"
subset = dataset[['ticker', 'user_location']]

city_list = ['Los Angeles CA', 'New York NY','San Francisco CA',
             'Washington DC','Tampa Fl', 'Dallas TX',
             'Newark NJ','Manteca CA','Austin TX',
             'Chicago IL']

def load_data_set():
    data_set = []
    for city in city_list:
        city_row = subset[subset['user_location'] == city]
        data_set.append(pd.unique(city_row['ticker']).tolist())
    return data_set

def create_C1(dataset):
    """
    Create frequent candidate 1-itemset C1 by scaning dataset.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
    Returns:
        C1: A set which contains all frequent candidate 1-itemsets
    """
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1

def is_apriori(Ck_item, Lksub1):
    """
    Judge whether a frequent candidate k-itemset satisfy Apriori property.
    Args:
        Ck_item: a frequent candidate k-itemset in Ck which contains all frequent
                 candidate k-itemsets.
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
    Returns:
        True: satisfying Apriori property.
        False: Not satisfying Apriori property.
    """
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_Ck(Lksub1, k):
    """
    Create Ck, a set which contains all all frequent candidate k-itemsets
    by Lk-1's own connection operation.
    Args:
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
        k: the item number of a frequent itemset.
    Return:
        Ck: a set which contains all all frequent candidate k-itemsets.
    """
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck


def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    """
    Generate Lk by executing a delete policy from Ck.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        Ck: A set which contains all all frequent candidate k-itemsets.
        min_support: The minimum support.
        support_data: A dictionary. The key is frequent itemset and the value is support.
    Returns:
        Lk: A set which contains all all frequent k-itemsets.
    """
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk

def generate_L(data_set, k, min_support):
    """
    Generate all frequent itemsets.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        k: Maximum number of items for all frequent itemsets.
        min_support: The minimum support.
    Returns:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and the value is support.
    """
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data

if __name__ == "__main__":
    """
    Test
    """
    data_set = load_data_set()
    L, support_data = generate_L(data_set, k=5, min_support=0.3)
    for Lk in L:
        print ("="*50)
        print ("frequent " + str(len(list(Lk)[0])) + "-itemsets\t\tsupport")
        print ("="*50)
        for freq_set in Lk:
            print (freq_set, support_data[freq_set])

    new_set = set()
    for LK in L:
        for LKK in LK:
            print("\n#######################")
            for compare_LK in L:
                for compare_LKK in compare_LK:
                    # Check if LKK is equal or a subset of compare_LKK
                    if(LKK.issubset(compare_LKK) and (LKK != compare_LKK)):
                        print(LKK, "is subset of ", compare_LKK)
                        confident = (support_data[compare_LKK]/support_data[LKK])
                        # print("confident of ", LKK, "to ", compare_LKK, "is: ",confident )
                        print("confidence of %s -> %s is: %f" % (LKK, compare_LKK, confident))