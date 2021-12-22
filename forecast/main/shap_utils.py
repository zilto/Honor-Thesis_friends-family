import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy

def gen_dict_mean_shap(model,X_train,X_test, model_name,K = 50 , sample_N =10,save_dict = "./shap_plots",output_sorted = False):
  filename = save_dict+"/" + model_name + ".png"
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  plt.clf()
  X_train_summary = shap.kmeans(X_train, K)
  explainer = shap.KernelExplainer(model.predict_proba,X_train_summary,link='logit')
  sampled_X_test = X_test.sample(sample_N)
  shap_values = explainer.shap_values(sampled_X_test) 
  mean_shap = np.zeros(shap_values[0].shape[-1],)
  shap_dict = dict()
  for numstates in range(len(shap_values)):
    mean_shap +=  np.mean(np.abs(shap_values[numstates]),axis = 0)
  
  for colno in range(len(X_test.columns)):
    shap_dict[X_test.columns[colno]] = mean_shap[colno]
  
  sorted_shap_dict = sorted(shap_dict.items(), key=lambda item: -item[1])
  shap.summary_plot(shap_values,X_test,max_display=X_train.shape[-1],show=False)
  #shap.force_plot(explainer.expected_value, shap_values[0], X_test)
  plt.savefig(filename)

  if output_sorted:
    return sorted_shap_dict
  else:
    return shap_dict

def gen_dict_mean_shap_no_plot(model,X_train,X_test, model_name,K = 50 , sample_N =10,output_sorted = False):
  filename = save_dict+"/" + model_name + ".png"
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  X_train_summary = shap.sample(X_train, K)
  explainer = shap.KernelExplainer(model.predict,X_train_summary)
  sampled_X_test = X_test.sample(sample_N)
  shap_values = explainer.shap_values(sampled_X_test) 
  mean_shap = np.zeros(shap_values[0].shape[-1],)
  shap_dict = dict()
  for numstates in range(len(shap_values)):
    mean_shap +=  np.mean(np.abs(shap_values[numstates]),axis = 0)
  
  for colno in range(len(X_test.columns)):
    shap_dict[X_test.columns[colno]] = mean_shap[colno]
  
  sorted_shap_dict = sorted(shap_dict.items(), key=lambda item: -item[1])

  if output_sorted:
    return sorted_shap_dict
  else:
    return shap_dict


def weighted_Kendall_tau(seq1,seq2):
  return scipy.stats.weightedtau(x, y, rank=True, weigher=None, additive=True)