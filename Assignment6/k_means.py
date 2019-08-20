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
from helper_package import assign_labels as al


def k_means_classify(df1, k):
    """
    Classify feature set of a DataFrame into k clusters
    :param df1: DataFrame with 'Mean_Return' and 'Std_Return' columns
    :param k: number of clusters
    """
    try:
        x = df1[['Mean_Return', 'Std_Return']].values
        kmeans_classifier = KMeans(n_clusters=k, init='random')
        y_means = kmeans_classifier.fit_predict(x)
        df1['cluster'] = y_means
    except KeyError as ke:
        print(ke)
        print('Mean_Return and/or Std_Return were not columns within input'
              'DataFrame')


def get_cluster_info(df1, k):
    """
    Gets label type percentages for each cluster and returns info in
    new DataFrame
    :param df1: DataFrame with 'cluster' column filled with k_means_classify()
    :param k: number of clusters used in k_means_classify
    :return: DataFrame with cluster label info
    """

    # create empty df for cluster label data
    cluster_df = pd.DataFrame(index={'Green', 'Red'})

    # iterate through clusters and calculate percentages for each label
    for i in range(0, k):

        column_name = 'cluster_' + str(i)

        try:
            # get cluster labels using cluster_value()
            cluster_labels = df1.apply(
                lambda a: cluster_value(a.binary_label, a.cluster, i), axis=1)

            # add cluster
            df1[column_name] = cluster_labels
        except KeyError as ke:
            print(ke)
            print('binary_label and/or cluster are not columns within input'
                  'DataFrame')

        # calculate pct of green and red values per cluster, ignores NaN values
        green_pct = (df1[column_name].sum() / df1[column_name].count()) * 100
        red_pct = 100 - green_pct

        # output percentages of each label
        print('\nCluster', i, 'color percentages:')
        print('Green: ' + str(round(green_pct, 4)) + '%')
        print('Red: ' + str(round(red_pct, 4)) + '%')

        pct_list = [green_pct, red_pct]  # list with green and red percentages
        cluster_df[column_name] = pct_list  # append values to cluster column

    # return transposed DataFrame with red and green percent values by cluster
    return cluster_df.T


def cluster_value(label, cluster, i):
    """
    Checks if i matches cluster value. Returns label value if it matches,
    otherwise it returns None.

    Used when iterating through list of cluster values.
    :param label: Label value
    :param cluster: cluster number
    :param i:
    :return:
    """
    if cluster == i:
        return label
    else:
        return None


def k_means_classify_multiple(df1, k_values):
    """
    Classify labels for df1 using multiple k values. Plot k value vs inertia
    and save plot as separate pdf file.
    :param df1: DataFrame with 'Mean_Return' and 'Std_Return' values
    :param k_values: List containing k values to be tested
    """
    x = df1[['Mean_Return', 'Std_Return']]  # feature set
    inertia_list = []  # list for inertia values (for each k)

    # iterate through k_values to get inertia value
    for k in k_values:
        kmeans_classifier = KMeans(n_clusters=k, init='random')
        y_means = kmeans_classifier.fit_predict(x)
        inertia = kmeans_classifier.inertia_
        inertia_list.append(inertia)

    # plot k_values vs inertia
    fig, ax = plt.subplots(1, figsize=(7, 5))
    plt.plot(range(min(k_values), max(k_values)+1),
             inertia_list, marker='o', color='green')
    plt.title('k-means cluster value vs inertia')
    plt.xlabel('number of clusters: k')
    plt.ylabel('inertia')
    plt.tight_layout()

    # get relative project directory and save figures to output file
    plot_dir = os.sep.join(os.path.dirname(os.path.realpath(__file__)).
                           split(os.sep)[:-2])

    # save plot to pdf
    plot_name = 'k_means_plot'
    output_file = os.path.join(
        plot_dir, 'CS677_Assignments', 'plots', plot_name + '.pdf')
    plt.savefig(output_file)


if __name__ == '__main__':

    # raises numpy errors/warnings so they can be caught by try/except
    np.seterr(all='raise')

    # allow df console output to display more columns
    hf.show_more_df()

    # get DataFrame of stock ticker info from csv file
    df = hf.fix_column_names(hf.get_ticker_df())

    df = al.assign_color_labels(df)  # assign color labels
    df = fs.get_feature_set(df)  # add mean and std return columns for DF

    k_list = [1, 2, 3, 4, 5, 6, 7, 8]  # list of k values to plot

    # create DataFrames for 2017 and 2018 from df
    df_2017_2018 = df.loc[df.Year.isin(['2017', '2018'])].reset_index()

    # feature set (mu, sigma) of df_2017_2018
    df_feature_set_data = df_2017_2018[
        ['Mean_Return', 'Std_Return', 'Color', 'binary_label']].groupby(
        df_2017_2018.Year_Week).max().reset_index()

    # ---------- Question 1 ----------
    print('\n__________Question 1__________')

    # classify using k-means (k=3) and do nothing with it
    k_means_classify(df_feature_set_data, 3)

    # classify using multiple k-means and find 'best' k value
    k_means_classify_multiple(df_feature_set_data, k_list)
    print('Best k value based on the "knee" method: 5')

    # ---------- Question 2 ----------
    print('\n__________Question 2__________')
    # classify using best k value, output label % for each cluster
    k_means_classify(df_feature_set_data, 5)

    # ---------- Question 3 ----------
    # get df of label percentages in each cluster
    cluster_colors = get_cluster_info(df_feature_set_data, 5)

    print('\n__________Question 3__________')
    print(cluster_colors)

    # get max red and green percentages
    max_red_pct = cluster_colors.Red.max()
    max_green_pct = cluster_colors.Green.max()

    if max_red_pct >= 90 or max_green_pct >= 90:
        print('\nk-means clustering found pure clusters')
    else:
        print('\nk-means clustering did not find pure clusters')
