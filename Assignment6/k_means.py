"""
James Sullivan
Class: CS677 - Summer 2
Date: 8/18/2019
Homework: k-means Questions #1-3

Implement k-means clustering. Your objects are weeks and your feature set is
(mu, sigma) for your weeks. We will choose the optimal number of clusters using
year 1 and year 2 data. Apply the classifier to your data and examine the
composition of your clusters.

1. Take k=3 and use k-means sklearn library routing for k-means (random
   initialization and use the defaults). *** We aren't asked to do anything
   with this information... why is it here? ***

   Take k=1, 2, ..., 7. 8 and compute
   the distortion vs. k. Use the "knee" method to find out the best k.
2. For this optimal k, examine your clusters and for each cluster, compute
   the percentage of "green" and "red" weeks in that cluster.
3. Does your k-means clustering find any "pure" clusters (percent of red or
   green weeks in a cluster is more than, say, 90%% of all weeks in that
   cluster)?
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from helper_package import helper_functions as hf
from helper_package import feature_set as fs
from helper_package import confusion_matrix_calcs as cm
from helper_package import assign_labels as al

# raises numpy errors/warnings so they can be caught by try/except
np.seterr(all='raise')

# allow df console output to display more columns
hf.show_more_df()

# get DataFrame of stock ticker info from csv file
df = hf.fix_column_names(hf.get_ticker_df())

df = al.assign_color_labels(df)  # assign color labels
df = fs.get_feature_set(df)      # add mean and std return columns for DF


k_list = [1, 2, 3, 4, 5, 6, 7, 8]


def k_means_classify(df1, k):

    x = df1[['Mean_Return', 'Std_Return']].values
    kmeans_classifier = KMeans(n_clusters=k, init='random')
    y_means = kmeans_classifier.fit_predict(x)
    df1['cluster'] = y_means


def get_cluster_info(df1, k):
    color_pct_dict = {}

    cluster_df = pd.DataFrame(index={'Green', 'Red'})

    for i in range(0, k):
        column_name = 'cluster_' + str(i)
        cluster_labels = df1.apply(lambda a: cluster_value(a.binary_label, a.cluster, i), axis=1)
        # df1[column_name] = df1.apply(lambda a: cluster_value(a.binary_label, a.cluster, i), axis=1)
        df1[column_name] = cluster_labels

        # calculate pct of green and red values per cluster, ignores NaN values
        green_pct = (df1[column_name].sum() / df1[column_name].count()) * 100
        red_pct = 100 - green_pct

        print('\nCluster', i, 'color percentages:')
        print('Green: ' + str(round(green_pct, 4)) + '%')
        print('Red: ' + str(round(red_pct, 4)) + '%')

        pct_list = [green_pct, red_pct]

        cluster_df[column_name] = pct_list

        color_pct_dict[column_name] = (green_pct, red_pct)

    # return transposed DataFrame with red and green percentage values by cluster
    return cluster_df.T


def cluster_value(label, cluster, i):

    if cluster == i:
        return label
    else:
        return None


def k_means_classify_multiple(df1, k_values):

    x = df1[['Mean_Return', 'Std_Return']]
    inertia_list = []

    for k in k_values:
        kmeans_classifier = KMeans(n_clusters=k, init='random')
        y_means = kmeans_classifier.fit_predict(x)
        inertia = kmeans_classifier.inertia_
        inertia_list.append(inertia)

    fig, ax = plt.subplots(1, figsize=(7, 5))
    plt.plot(range(min(k_values), max(k_values)+1), inertia_list, marker='o', color='green')
    plt.title('k-means cluster value vs inertia')
    plt.xlabel('number of clusters: k')
    plt.ylabel('inertia')
    plt.tight_layout()

    #plt.show()

    # get relative project directory and save figures to output file
    plot_dir = os.sep.join(os.path.dirname(os.path.realpath(__file__)).
                           split(os.sep)[:-2])

    plot_name = 'k_means_plot'
    output_file = os.path.join(
        plot_dir, 'CS677_Assignments', 'plots', plot_name + '.pdf')
    plt.savefig(output_file)


# create DataFrames for 2017 and 2018 from df
df_2017_2018 = df.loc[df.Year.isin(['2017', '2018'])].reset_index()

df_feature_set_data = df_2017_2018[
    ['Mean_Return', 'Std_Return', 'Color', 'binary_label']].groupby(
    df_2017_2018.Year_Week).max().reset_index()



if __name__ == '__main__':

    # ---------- Question 1 ----------
    print('\n__________Question 1__________')
    k_means_classify(df_feature_set_data, 3)
    k_means_classify_multiple(df_feature_set_data, k_list)
    print('Best k value based on the "knee" method: 5')

    # ---------- Question 2 ----------
    print('\n__________Question 2__________')
    k_means_classify(df_feature_set_data, 5)

    # ---------- Question 3 ----------
    cluster_colors = get_cluster_info(df_feature_set_data, 5)

    print('\n__________Question 3__________')
    print(cluster_colors)

    max_red_pct = cluster_colors.Red.max()
    max_green_pct = cluster_colors.Green.max()

    if max_red_pct >= 90 or max_green_pct >= 90:
        print('\nk-means clustering found pure clusters')
    else:
        print('\nk-means clustering did not find pure clusters')
