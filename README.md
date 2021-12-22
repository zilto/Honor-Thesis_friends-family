# Friends and Family
## Overview
This repository contains work from my undergrad Honor thesis, realized under the supervision of Prof. Pierre Orban, and with the support of Rose Guay Hottin and Shivam Patel. The goal of the project was to predict the "next-day" self-reported levels of stress and happiness based on smartphone sensors. This includes measurements and signals from: Wi-Fi, Bluetooth, GPS, accelerometer, application usage, battery metadata, SMS logs, and call logs. This approach is part of the novel field of *Digital phenotyping* and aims to enable better monitoring of psychiatric conditions and adopt preventive actions. The project was realized using the Friends&Family dataset (http://realitycommons.media.mit.edu/friendsdataset.html).

## Content of the repository
- ``preprocessing.py`` contains the pipeline to programmatically extract and tidy the dataset, and then preprocess features for the learning models
- ``features_exploration.ipynb`` and ``label_exploration.ipynb`` contain some data exploration of the features and labels
- ``LSTM_8h_7days.ipynb`` and ``XGBoost_30min_1day.ipynb`` are used to train LSTM and XGBoost models
- ``statistical_analysis.ipynb`` contains the analysis of models' performance, the feature importance, and the figures
- ``/report/honor-report_JEAN-Thierry.pdf`` is the submission for my honor thesis
- ``/report/octobre_IVADO_phenotypage_digital_TJ.pdf`` is the slide deck for a scientific communication I did in October 2021

## What I learned
- Research pipelining and reproducible data science (I improved since this project!)
- Data wrangling
- Time series analysis and formatting using pandas
