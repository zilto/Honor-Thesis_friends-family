import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import sys
np.set_printoptions(threshold=sys.maxsize)

from sklearn import metrics
from sklearn.preprocessing import PowerTransformer
# import missingno as msno

np.random.seed(123)


def create_same_length_instances(df,length, label, timestamp_name):
    df = df.drop(columns=[timestamp_name])
    full_data = pd.DataFrame()
    indexes = list(df[df[label].notnull()].index) # Choose label indexes
    start = 0
    for i in indexes:
        if i - start >= length:
            # If the length of instance is higher than treshold length,
            # It chooses last timestamps (according to given length)
            instance = df.iloc[i-length+1:i+1, :]
            full_data = full_data.append(instance, ignore_index=True, sort=False)
        else:
            # If the length of instance is lower than treshold length,
            # It chooses all timestamps and add zeros to head timestamps until reach given length.
            index_diff = i - start
            instance = df.iloc[i-index_diff+1:i+1, :]
            back_fill = np.empty((length-index_diff, df.shape[1]))
            back_fill.fill(np.nan)
            back_fill = pd.DataFrame(back_fill, columns=df.columns)
            instance = back_fill.append(instance, ignore_index=True, sort=False)
            full_data = full_data.append(instance, ignore_index=True, sort=False)
        start = i
    return full_data


def normalize_data(df, label):
	"""
	Normalises the data from the same lenght instance outputs
	df : The dataframe
	label : The name of the target label which could be STRESS/MOOD etc
	timestamp_name : The name of column in data frame with has the timestamp
	"""
	df_same_labels = df[label].copy()
	#df_time_labels = df[timestamp_name].copy()
	df = df.drop(columns=[label]) # , timestamp_name
	df_columns = df.columns
	pt = PowerTransformer(method='yeo-johnson', standardize=True)
	pt.fit(df)
	df_norm = pt.transform(df)
	df_norm = pd.DataFrame(df_norm, columns=df_columns)
	df_norm[label] = df_same_labels
	#df_norm[timestamp_name] = df_time_labels

	return df_norm


def fill_nulls(df, label_col, method="mean"):
    if method == 0:
        df.loc[:, df.columns != label_col] = df.loc[:, df.columns != label_col].fillna(0)
    elif method == 'mean':
        df.loc[:, df.columns != label_col] = df.loc[:, df.columns != label_col].fillna(df.loc[:, df.columns != label_col].mean())
    return df


def create_instances_dl(df, length, label):
    total_instances = df[df[label].notnull()].shape[0]
    total_features = df[df[label].notnull()].shape[1] - 1
    indexes = list(df[df[label].notnull()].index)
    data = df.drop(columns=[label])
    all_data = np.empty(shape=(total_instances, length, total_features))
    for instance_no, label_index in enumerate(indexes):
        start = label_index - length + 1
        all_data[instance_no] = data.iloc[start:label_index+1, :].values
    return all_data


def create_instances_ml(df, length, label):
    indexes = list(df[df[label].notnull()].index)
    data = df.drop(columns=[label])
    all_data = pd.DataFrame()
    for i in indexes:
        start = i-length+1
        sample = data.iloc[start:i, :]
        stats_dict = {}
        for j in sample.columns:
            stats_dict[j + '_mean'] = sample[j].mean()
            stats_dict[j + '_median'] = sample[j].median()
            stats_dict[j + '_min'] = sample[j].min()
            stats_dict[j + '_max'] = sample[j].max()
            stats_dict[j + '_std'] = sample[j].std()
            stats_dict[j + '_skew'] = sample[j].skew()
        stats_dict['STRESSED'] = df.loc[i, label]
#         new_data = pd.DataFrame.from_dict(stats_dict)
        all_data = all_data.append(pd.DataFrame(stats_dict, index=[0]), ignore_index=True, sort=False)
    return all_data


def create_random_samples(X, y, train_size=1000, seed=1, balanced_test=False):
    random.seed(seed)
    one_class_size = int(train_size / 2)
    all_one_indexes = [i for i, x in enumerate(y) if x == 1]
    all_zero_indexes = [i for i, x in enumerate(y) if x == 0]
    ones = random.sample(all_one_indexes, one_class_size)
    zeros = random.sample(all_zero_indexes, one_class_size)
    train_indexes = random.sample(ones+zeros, train_size)
    new_x = X[train_indexes]
    new_y = y[train_indexes]
    if balanced_test == False:
        test_indexes = [i for i in range(len(y)) if i not in train_indexes]
        test_x = X[train_indexes]
        test_y = y[train_indexes]
    else:
        not_used_ones = list(set(all_one_indexes) - set(ones))
        not_used_zeros = list(set(all_zero_indexes) - set(zeros))
        test_sample_size = min(len(not_used_ones), len(not_used_zeros))
        test_ones = random.sample(not_used_ones, test_sample_size)
        test_zeros = random.sample(not_used_zeros, test_sample_size)
        test_indexes = random.sample(test_ones+test_zeros, test_sample_size*2)
        test_x = X[test_indexes]
        test_y = y[test_indexes]
    
    return new_x, new_y, test_x, test_y