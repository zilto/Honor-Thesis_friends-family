import os
import re

import numpy as np
import pandas as pd

from config import DATA_DIR, COLS_DTYPES


def _match_userid(fname):
    # regex to identify file by userid
    r = r"((sp)|(fa))\d{2}-\d{2}-\d{2}"
    return re.search(r, fname).group(0)


def _try_create_dir(loc, dir_name):
    # creates output folder if it does not exist
    path = os.path.join(loc, dir_name)
    try:
        os.makedirs(path)
    except:
        pass
    return path


def _get_feature(feature, userid):
    # if userid is not specified, get the file containing all users
    path = os.path.join(DATA_DIR, f"original/{feature}")
        
    try:
        feature = pd.read_csv(path + f"/{feature}_{userid}.csv", index_col=False, dtype=COLS_DTYPES[feature])
    except:
        print(f"{feature} file missing for {userid}")
        feature = pd.DataFrame(columns=["timestamp", "userid"])
    
    feature.timestamp = pd.to_datetime(feature.timestamp)
    feature = feature.set_index("timestamp")
    return feature


def _merge_features(features, userid):
    merged_df = None
    for feature in features:
        if merged_df is None:
            merged_df = _get_feature(feature, userid)
        else:    
        # the only shared column names between features are ["timestamp", "userid"]
        # when getting features from their .csv, "timestamp" is converted to datetime;see get_feature()
            df = _get_feature(feature, userid)
            merged_df = pd.merge(merged_df, df, on=["timestamp", "userid"], how="outer")    
    # replaces numpy infinity values by NaN
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    return merged_df


def _downcast_dtypes(df):
    """Downcasts DataFrame columns to float32 (except datetime and object dtypes)
    
    Float32 reduces memory usage and accepts NaN values, 
    
    Args:
        df (DataFrame): dataframe to be reduced

    Returns:
        df (DataFrame): reduced dataframe
    """
    
    _start = df.memory_usage(deep=True).sum() / 1024 ** 2
    for c in df:
        if df[c].dtype not in ["datetime64[ns]", "object"]:
            df[c] = df[c].astype(np.float32)
    _end = df.memory_usage(deep=True).sum() / 1024 ** 2
    saved = (_start - _end) / _start * 100
    return df


def _load_userdf(userid, SR="RAW"):
    """Loads csv file of preprocessed data for a given user and sample rate
    
    Args:
        userid (str): user id (or participantID); also found in config.py
        SR (str, optional): sample rate of the preprocessed and resampled data. "RAW" loads data at original sample rate.

    Returns:
        userdf (DataFrame): DataFrame with user data
    """
    
    userdf = pd.read_csv(DATA_DIR + f"/per_user/{SR}/{userid}_{SR}_data.csv")
    return userdf