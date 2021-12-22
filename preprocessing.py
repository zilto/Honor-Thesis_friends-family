import os
import zipfile
import tarfile
import json

import numpy as np
import pandas as pd

from utils import _match_userid, _try_create_dir, _get_feature, _downcast_dtypes
from .config import DATA_DIR, COLS_NAMES, COLS_DTYPES, USERIDS


### INITIAL DATA ACQUISITION AND FILE SPLITTING FUNCTIONS BEGIN ###

def decompress_dataset():
    """ Decompress the original dataset .zip file and creates directory
    """
    path = os.path.join(DATA_DIR, "Friends&Family.zip")
    
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)


def extract_features_per_user(include_all=False):
    """ Separates original files into .csv per user 
    """
    
    # path to original dataset
    og_path = os.path.join(DATA_DIR, "DataRelease/") 
    
    # read entries in the original dataset directory
    with os.scandir(og_path) as it:
        for entry in it:
            # check if entry is a file and has .bz2 extension
            if entry.is_file() and entry.name.endswith(".bz2"):
                feature_name = entry.name.partition(".")[0].strip()
                feature_cols = COLS_NAMES[feature_name]
                cols_dtypes = COLS_DTYPES[feature_name]
                df = pd.read_csv(entry.path, header=0, names=feature_cols, dtype=cols_dtypes, parse_dates=["timestamp"],
                                 encoding="latin-1", compression="bz2")
                # drop column on "userid_B" when in contact with another study participant
                if feature_name in ["BluetoothProximity", "CallLog", "SMSLog"]:
                    df = df.drop(columns=["userid_B"])           
                
                dest_path = _try_create_dir(DATA_DIR, f"original/{feature_name}")
                if include_all:
                    df.to_csv(dest_path + f"/{feature_name}_all-users.csv", index=False)
                
                for userid in USERIDS: # this format was preferred to "groupby" to make sure a DataFrame is created for each user, even if empty
                    userdf = df.loc[df["userid"] == userid]
                    userdf.to_csv(dest_path + f"/{feature_name}_{userid}.csv", index=False)
                print(f"{feature_name} extracted for all users")


def extract_wifi_per_user():  
    """ Extracts original wifi files from tarfile
    """
    
    og_path = os.path.join(DATA_DIR, "DataRelease/")
    dest_path = _try_create_dir(DATA_DIR, "original/WiFi")
    
    wifi_tar = tarfile.open(og_path + "WiFi.tar")
    wifi_tar.extractall(og_path) 
    wifi_path = os.path.join(og_path, "CSV/")
    
    with os.scandir(wifi_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.startswith("wlan"):
                feature_cols = COLS_NAMES["WiFi"]
                userdf = pd.read_csv(entry.path, header=0, names=feature_cols,
                                     encoding="latin-1", compression="bz2")
                userid = _match_userid(entry.name)
                userdf.to_csv(dest_path + f"/WiFi_{userid}.csv", index=False)
                
    # file is missing for fa10-01-84
    userdf_84 = pd.DataFrame(columns=feature_cols)
    userdf_84.to_csv(dest_path + "/WiFi_fa10-01-84.csv", index=False)
    
    print("WiFi extracted for all users")


def extract_ema_per_user():
    """ Parse daily ema surveys from SurveyFromPhone.csv
    """
    og_path = os.path.join(DATA_DIR, "DataRelease/")
    dest_path = _try_create_dir(DATA_DIR, "original/DailyEMA")
    
    feature_cols = COLS_NAMES["SurveyFromPhone"]
    df = pd.read_csv(og_path + "SurveyFromPhone.csv.bz2", header=0, names=feature_cols,
                     encoding="latin-1", compression="bz2")  
    
    for userid in USERIDS: # this format was preferred to "groupby" to make sure a DataFrame is created for each user, even if empty
        userdf = df.loc[df["userid"] == userid] 
        
        filtered = userdf[userdf["survey_name"].str.contains("day")]
        # dictionary of {timestamp : daily EMA}; EMA in JSON format
        survey_dict = dict(zip(filtered.timestamp, filtered.answers_raw))

        # unpacks EMA from JSON;
        answer_dict = {}
        for week, ema in survey_dict.items():
            # load JSON and select the EMA answers
            answers = json.loads(ema)["answerData"][0]
            # valid if JSON contains 6 values
            if len(answers) == 6:
                answer_dict[week] = answers

        answer_df = pd.DataFrame.from_dict(answer_dict,
                                           columns=["happy", "stress", "productive",
                                                    "eat_healthy", "sleep_h", "social_h"],
                                           orient="index")
        # merge with the "ema date", different from survey completed date (aka timestamp)
        answer_merged = pd.merge(answer_df, userdf[["timestamp", "survey_name"]],
                                 left_index=True, right_on="timestamp", how="left")
        # parse string to datetime and normalize to 12:00:00
        answer_merged["survey_name"] = pd.to_datetime(answer_merged["survey_name"],
                                                      format="%A %m/%d/%y").dt.normalize()
        answer_merged = answer_merged.rename(columns={"survey_name":"ema_date"})
        # replaces non integer values
        answer_merged = answer_merged.replace({"<1": 0, ">3": 4, "<5": 4, ">9": 10})
        # fill a column with userid
        answer_merged["userid"] = userid
        answer_merged.to_csv(dest_path + f"/DailyEMA_{userid}.csv", index=False)

   
def merge_all_features_per_user():
    dest_path = _try_create_dir(DATA_DIR, "per_user")
    
    for userid in USERIDS:
        merged_df = None
        for feature, cols in COLS_NAMES.items():
            if merged_df is None:
                merged_df = _get_feature(feature, userid)
            else:    
            # the only shared column names between features are ["timestamp", "userid"]
            # when getting features from their .csv, "timestamp" is converted to datetime;see get_feature()
                df = _get_feature(feature, userid)
                merged_df = pd.merge(merged_df, df, on=["timestamp", "userid"], how="outer")    
        # replaces numpy infinity values by NaN
        merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
        print(f"Shape of df after merge: {merged_df.shape}")
        merged_df.to_csv(dest_path + f"/{userid}_RAW_data.csv", index=False)
        print(f"Merging for {userid} is completed.")
    print("All users merged")
    
### INITIAL DATA ACQUISITION AND FILE SPLITTING FUNCTIONS END ###

    
### DATA READ AND FEATURE PREPROCESSING FUNCTIONS BEGIN ###

def accel_norm(userid):
    accel_df = _get_feature("Accel", userid)
    accel_df["accel_norm"] = np.linalg.norm(
        np.array([accel_df["accel_x"], accel_df["accel_y"], accel_df["accel_z"]]),
        axis=0
        )
    accel_df = accel_df.drop(columns=["accel_x", "accel_y", "accel_z"])
    return accel_df

def activity_ratio(userid):
    accel_accum = _get_feature("AccelAccum", userid)
    
    accel_accum["accel_activity_interval_duration"] = accel_accum["accel_activity_interval_duration"].map({15000: 16, 20000: 21})
    accel_accum["accel_total_activity_count"] = accel_accum["accel_low_activity_count"] + accel_accum["accel_high_activity_count"]
    
    # calculates ratio according to interval duration
    accel_accum["low_activity_ratio"] = accel_accum["accel_low_activity_count"] / accel_accum["accel_activity_interval_duration"]
    accel_accum["high_activity_ratio"] = accel_accum["accel_high_activity_count"] / accel_accum["accel_activity_interval_duration"]
    accel_accum["total_activity_ratio"] = accel_accum["accel_total_activity_count"] / accel_accum["accel_activity_interval_duration"]
    
    return accel_accum


def app_installed_count(userid):
    app_installed = _get_feature("App", userid)  
    
    # convert app package name to category
    app_installed["app_package"] = app_installed["app_package"].astype("category").cat.codes
    # count app package per timestamp (per scan)
    app_installed = app_installed.groupby("timestamp").count()
    app_installed = app_installed.rename(columns={"app_package": "app_installed_count"})
    app_installed = app_installed.drop(columns=["app_uninstalled"])
    app_installed["userid"] = userid
    
    return app_installed


def app_running_count(userid):
    app_running = _get_feature("AppRunning", userid)
    
    app_running["app_running_package"] = app_running["app_running_package"].astype("category").cat.codes
    app_running = app_running.groupby("timestamp").count()
    app_running = app_running.rename(columns={"app_running_package": "app_running_count"})
    app_running = app_running.drop(columns=["app_running_class", "app_running_topclass"])
    app_running["userid"] = userid
    
    return app_running


def battery_status(userid):
    battery = _get_feature("Battery", userid)

    # binarize feature by keeping the only relevant category
    battery["battery_is_plugged"] = (battery["battery_plugged"] != 0).astype(int)
    battery["battery_is_discharging"] = (battery["battery_status"] == 3).astype(int)
    battery = battery.drop(columns=["battery_techno", "battery_present", "battery_health",
                                   "battery_status", "battery_plugged"],
                          )
    
    return battery


def bluetooth_contact_count(userid):
    bluetooth = _get_feature("BluetoothProximity", userid)
    
    bluetooth["bt_mac_addr"] = bluetooth["bt_mac_addr"].astype("category").cat.codes
    bluetooth = bluetooth.groupby("timestamp").count()
    bluetooth = bluetooth.rename(columns={"bt_mac_addr": "bt_mac_addr_count"})
    bluetooth["userid"] = userid
    
    return bluetooth


def call_count(userid):
    call = _get_feature("CallLog", userid)

    # rename values
    call["call_type"] = call["call_type"].replace({"incoming+": "incoming",
                                                   "outgoing+": "outgoing"})
    # create dummies / OneHot column encoding
    call = pd.get_dummies(call, columns=["call_type"])
    call["call_phone_hash"] = call["call_phone_hash"].astype("category").cat.codes
    
    return call


def location_pairwise_dist(userid):
    loc_df = _get_feature("Location", userid)
    
    pairwise_diff = np.diff(loc_df[["loc_x", "loc_y"]].values, axis=0)
    pairwise_dists = np.sqrt((pairwise_diff** 2).sum(axis=1))
    loc_df["loc_pairwise_dist"] = np.insert(pairwise_dists, 0, np.nan)
    
    return loc_df


def location_dist_from_og(userid):
    loc_df = _get_feature("Location", userid)
    
    # euclidian distance between the relative (0, 0) and the current location
    loc_df["loc_dist_from_og"] = np.sqrt(loc_df["loc_x"].pow(2) + loc_df["loc_y"].pow(2))
    
    return loc_df


def sms_count(userid):
    sms = _get_feature("SMSLog", userid)
    
    sms = pd.get_dummies(sms, columns=["sms_type"]) # create dummies / OneHot column encoding 
    sms["sms_phone_hash"] = sms["sms_phone_hash"].astype("category").cat.codes
    
    return sms


def wifi_count(userid):
    wifi = _get_feature("WiFi", userid)    
    
    wifi["wifi_scan_nearer"] = np.where(
        np.logical_and(wifi["wifi_rssi"] >= -65, wifi["wifi_rssi"] <= 0),
        1, 0)
    wifi["wifi_scan_near"] = np.where(
        np.logical_and(wifi["wifi_rssi"] >= -80, wifi["wifi_rssi"] < -65),
        1, 0)
    wifi["wifi_scan_far"] = np.where(
        np.logical_and(wifi["wifi_rssi"] >= -90, wifi["wifi_rssi"] < -80),
        1, 0)
    wifi["wifi_scan_farther"] = np.where(
        np.logical_and(wifi["wifi_rssi"] >= -125, wifi["wifi_rssi"] < -90),
        1, 0)  # Normally -100 is max but for one anomaly.
    wifi["wifi_scan_total"] = 1
    wifi["wifi_rssi_avg"] = wifi["wifi_rssi"]
    wifi["wifi_rssi_std"] = wifi["wifi_rssi"]
    
    wifi = wifi.groupby("timestamp").agg(
        {"wifi_scan_nearer": np.sum,
        "wifi_scan_near": np.sum,
        "wifi_scan_far": np.sum,
        "wifi_scan_farther": np.sum,
        "wifi_scan_total": np.sum,
        "wifi_rssi_avg" : np.mean,
        "wifi_rssi_std" : np.std
        }
    )
    wifi["userid"] = userid
    return wifi


def wifi_rssi(userid):
    wifi = _get_feature("WiFi", userid)
    
    wifi["wifi_rssi_avg"] = wifi["wifi_rssi"]
    wifi["wifi_rssi_std"] = wifi["wifi_rssi"]
    
    wifi = wifi.groupby("timestamp").agg(
        {"wifi_rssi_avg" : np.mean,
         "wifi_rssi_std" : np.std
        }
    )
        
    wifi["userid"] = userid
    return wifi


def ema(userid):
    # the timestamp found in the file is the time of survey completion
    # the EMA date is the date asked about in the survey
    # set the EMA date as the new "timestamp" to align within predictive model
    ema = _get_feature("DailyEMA", userid)
    if ema.empty:
        return ema
    
    ema = ema.reset_index()
    ema["ema_date"] = pd.to_datetime(ema["ema_date"])
    ema = ema.rename(columns={"timestamp":"ema_completion_timestamp"})
    ema["ema_completion_hour"] = ema["ema_completion_timestamp"].dt.hour
    ema["ema_completion_offset"] = (ema["ema_completion_timestamp"] - ema["ema_date"]).dt.days
    ema = ema.drop(columns="ema_completion_timestamp")
    ema = ema.set_index("ema_date") 
    ema.index = ema.index.rename("timestamp")
    return ema

### DATA READ FUNCTIONS END ###


### MERGING AND RESAMPLING FUNCTIONS BEGIN ###

def binarize_ema(df):
    # any df with the ema columns can be passed
    # dictionary to binarize each ema
    labels = {
        "happy"       : {0: [1, 2, 3, 4],
                         1: [5, 6, 7]},
        "stress"      : {0: [1, 2, 3, 4],
                         1: [5, 6, 7]},
        "productive"  : {0: [1, 2, 3],
                         1: [4, 5, 6, 7]},
        "eat_healthy" : {0: [1, 2, 3],
                         1: [4, 5, 6, 7]},
        "sleep_h"     : {0: [4, 5, 6, 7],
                         1: [8, 9, 10]},
        "social_h"    : {0: [0],
                         1: [1, 2, 3, 4]}
    }

    for label, values in labels.items():
        df[label] = df[label].replace(values[0], 0)
        df[label] = df[label].replace(values[1], 1)
    return df


def resample_agg(df, SR="30min", origin="start"):
    if df.index.name != "timestamp":
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)
    
    # takes agg_dict (imported from config.py) and only keeps columns found in df
    cols_to_agg = {col: agg for col, agg in agg_dict.items() if col in df.columns}
    # resamples df at sample rate (SR) with aggregations methods passed in cols_to_agg
    df = df.resample(SR, origin=origin).agg(cols_to_agg)
    return df

### MERGING AND RESAMPLING FUNCTIONS END ###


### ROUTINES ###

def resample_all(SR="30min", origin="start"):
    path = _try_create_dir(DATA_DIR, f".output/{SR}")
    for userid in userid_list:
        try:
            userdf = pd.read_csv(DATA_DIR + f"per_user/.output/RAW/{userid}_RAW_data.csv")
            userdf = _downcast_dtypes(userdf)
            agg = resample_agg(userdf, SR, origin)
            agg["userid"] = userid
            agg.to_csv(path + f"/{userid}_{SR}_data.csv")
            print(f"Resampling for {userid} is completed.")
        except FileNotFoundError:
            print(f"{userid} file missing")
    print("All users resampled at: ", SR)


def append_all_users(SR="30min"):
    path = DATA_DIR + f"per_user/.output/{SR}"
    df = pd.read_csv(path + f"/{userid_list[0]}_{SR}_data.csv")
    df = _downcast_dtypes(df)
    for userid in userid_list:
        try:
            new_df = pd.read_csv(path + f"/{userid}_{SR}_data.csv")
            new_df = _downcast_dtypes(new_df)
            df = df.append(new_df)
        except FileNotFoundError:
            print(f"{userid} file missing")
    df.to_csv(path + f"/.combinedUsers_{SR}_data.csv")
    print("combinedUsers df shape: ", df.shape)
    print("Merge of all users completed")


def main():
    decompress_dataset()
    extract_features_per_user()
    extract_wifi_per_user()
    extract_ema_per_user()
    merge_all_features_per_user()
    
if __name__ == "__main__":
    main()