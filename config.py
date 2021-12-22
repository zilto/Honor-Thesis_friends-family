import os
import numpy as np

# set directory of unzipped dataset
DATA_DIR = "D:/.coding/datasets/Friends&Family/DataRelease/"
STUDY_DIR = "./studies/"


# dictionary of column names to assign when loading feature files
COLS_NAMES = {
    "Accel"              : ["userid", "timestamp", "accel_x", "accel_y",
                            "accel_z"],
    "AccelAccum"         : ["userid", "timestamp", "accel_low_activity_count",
                            "accel_high_activity_count", "accel_activity_interval_duration"],
    "App"                : ["userid", "timestamp", "app_package",
                            "app_uninstalled"],
    "AppRunning"         : ["userid", "timestamp", "app_running_package",
                            "app_running_class", "app_running_topclass"],
    "Battery"            : ["userid", "timestamp", "battery_level",
                            "battery_present", "battery_health", "battery_plugged",
                            "battery_status", "battery_techno", "battery_temp",
                            "battery_volt"],
    "BluetoothProximity" : ["userid", "timestamp", "userid_B",
                            "bt_mac_addr"],
    "CallLog"            : ["userid", "userid_B", "timestamp",
                            "call_type", "call_duration", "call_phone_hash"],
    "Location"           : ["userid", "timestamp", "loc_accuracy", "loc_x",
                            "loc_y"],
    "SMSLog"             : ["userid", "userid_B", "timestamp",
                            "sms_type", "sms_phone_hash"], 
    "SurveyFromPhone"    : ["userid", "timestamp", "survey_name",
                            "questions_raw", "answers_raw"],
    "WiFi"               : ["userid", "timestamp", "wifi_mac_addr", "wifi_rssi"]
    }


COLS_DTYPES = {
    "Accel": {
        "userid": "string",
        "accel_x": "float32",
        "accel_y": "float32",
        "accel_z": "float32"
    },

    "AccelAccum": {
        "userid": "string",
        "accel_low_activity_count": "float32",
        "accel_high_activity_count": "float32",
        "accel_activity_interval_duration": "float32"
    },

    "App": {
        "userid": "string",
        "app_package": "string",
        "app_uninstalled": "string"
    },

    "AppRunning": {
        "userid": "string",
        "app_running_package": "string",
        "app_running_class": "string",
        "app_running_topclass": "string"
    },

    "Battery": {
        "userid": "string",
        "battery_level": "float32",
        "battery_present": "float32",
        "battery_health": "float32",
        "battery_plugged": "float32",
        "battery_status": "float32",
        "battery_techno": "string",
        "battery_temp": "float32",
        "battery_volt": "float32",
    },

    "BluetoothProximity" : {
        "userid": "string",
        "userid_B": "string",
        "bt_mac_addr": "float32"
    },
    
    "CallLog" : {
        "userid": "string",
        "userid_B": "string",
        "call_type": "string",
        "call_duration": "float32",
        "call_phone_hash": "string"
    },
    
    "Location": {
        "userid": "string",
        "loc_accuracy": "float32",
        "loc_x": "float32",
        "loc_y": "float32"
    },
    
    "SMSLog" : {
        "userid": "string",
        "userid_B": "string",
        "sms_type": "string",
        "sms_phone_hash": "string"
    },

    "SurveyFromPhone": {
        "userid": "string",
        "survey_name": "string",
        "questions_raw": "object",
        "answers_raw": "object"
    },
    
    "WiFi": {
        "userid": "string",
        "wifi_mac_addr": "float32",
        "wifi_rssi": "float32"
    }
}


# dictionary of aggregation methods to use for each feature when resampling
AGG_DICT = {
    "accel_x": np.mean, 
    "accel_y": np.mean,
    "accel_z": np.mean,
    "accel_low_activity_count": np.sum,
    "accel_high_activity_count": np.sum,
    "accel_total_activity_count": np.sum,
    "accel_activity_interval_duration": np.sum,
    "app_running_count": np.max,
    "battery_temp": np.mean,
    "battery_volt": np.mean,
    "battery_level": np.mean,
    "battery_is_plugged": np.sum,
    "battery_is_discharging": np.sum,
    "bt_mac_addr_count": np.sum,
    "call_duration": np.sum,
    "call_phone_hash": "count",
    "call_type_incoming": np.sum,
    "call_type_missed": np.sum,
    "call_type_outgoing": np.sum,
    "loc_accuracy": np.mean,
    "loc_x": np.mean,
    "loc_y": np.mean,
    "sms_phone_hash": "count",
    "sms_type_incoming": np.sum,
    "sms_type_outgoing": np.sum,
    "wifi_scan_total": np.max,
    "wifi_scan_nearer": np.max,
    "wifi_scan_near": np.max,
    "wifi_scan_far": np.max,
    "wifi_scan_farther": np.max,
    "wifi_rssi_avg": np.max,
    "wifi_rssi_std": np.max,
    "ema_completion_offset": np.max,
    "ema_completion_hour": np.max,
    "happy": np.max,
    "stress": np.max,
    "productive": np.max,
    "eat_healthy": np.max,
    "sleep_h": np.max,
    "social_h": np.max
}


# list of all users' id 
USERIDS = [
    'fa10-01-01',
    'fa10-01-02',
    'fa10-01-03',
    'fa10-01-04',
    'fa10-01-05',
    'fa10-01-06',
    'fa10-01-07',
    'fa10-01-08',
    'fa10-01-09',
    'fa10-01-10',
    'fa10-01-11',
    'fa10-01-12',
    'fa10-01-13',
    'fa10-01-14',
    'fa10-01-15',
    'fa10-01-16',
    'fa10-01-17',
    'fa10-01-18',
    'fa10-01-19',
    'fa10-01-20',
    'fa10-01-21',
    'fa10-01-22',
    'fa10-01-23',
    'fa10-01-24',
    'fa10-01-25',
    'fa10-01-26',
    'fa10-01-27',
    'fa10-01-28',
    'fa10-01-29',
    'fa10-01-30',
    'fa10-01-31',
    'fa10-01-32',
    'fa10-01-33',
    'fa10-01-34',
    'fa10-01-35',
    'fa10-01-36',
    'fa10-01-37',
    'fa10-01-38',
    'fa10-01-39',
    'fa10-01-40',
    'fa10-01-41',
    'fa10-01-42',
    'fa10-01-43',
    'fa10-01-44',
    'fa10-01-45',
    'fa10-01-46',
    'fa10-01-47',
    'fa10-01-48',
    'fa10-01-49',
    'fa10-01-50',
    'fa10-01-51',
    'fa10-01-52',
    'fa10-01-53',
    'fa10-01-54',
    'fa10-01-55',
    'fa10-01-56',
    'fa10-01-57',
    'fa10-01-58',
    'fa10-01-59',
    'fa10-01-60',
    'fa10-01-61',
    'fa10-01-62',
    'fa10-01-63',
    'fa10-01-64',
    'fa10-01-65',
    'fa10-01-66',
    'fa10-01-67',
    'fa10-01-68',
    'fa10-01-69',
    'fa10-01-70',
    'fa10-01-71',
    'fa10-01-72',
    'fa10-01-73',
    'fa10-01-74',
    'fa10-01-75',
    'fa10-01-76',
    'fa10-01-77',
    'fa10-01-78',
    'fa10-01-79',
    'fa10-01-80',
    'fa10-01-81',
    'fa10-01-82',
    'fa10-01-83',
    'fa10-01-84',
    'fa10-01-85',
    'fa10-01-86',
    'sp10-01-01',
    'sp10-01-02',
    'sp10-01-03',
    'sp10-01-04',
    'sp10-01-05',
    'sp10-01-06',
    'sp10-01-07',
    'sp10-01-08',
    'sp10-01-09',
    'sp10-01-10',
    'sp10-01-11',
    'sp10-01-12',
    'sp10-01-13',
    'sp10-01-14',
    'sp10-01-15',
    'sp10-01-16',
    'sp10-01-17',
    'sp10-01-18',
    'sp10-01-19',
    'sp10-01-20',
    'sp10-01-21',
    'sp10-01-22',
    'sp10-01-23',
    'sp10-01-24',
    'sp10-01-25',
    'sp10-01-26',
    'sp10-01-27',
    'sp10-01-28',
    'sp10-01-29',
    'sp10-01-30',
    'sp10-01-31',
    'sp10-01-32',
    'sp10-01-33',
    'sp10-01-34',
    'sp10-01-35',
    'sp10-01-36',
    'sp10-01-37',
    'sp10-01-38',
    'sp10-01-39',
    'sp10-01-40',
    'sp10-01-41',
    'sp10-01-42',
    'sp10-01-43',
    'sp10-01-44',
    'sp10-01-45',
    'sp10-01-46',
    'sp10-01-47',
    'sp10-01-48',
    'sp10-01-49',
    'sp10-01-50',
    'sp10-01-51',
    'sp10-01-52',
    'sp10-01-53',
    'sp10-01-54',
    'sp10-01-55',
    'sp10-01-56'
    ]