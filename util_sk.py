#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:54:37 2020

@author: osita
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
#######################################################################
#test with scikit learn accuracy
# Use sklearn to compute auc
def compMetrics(p_val, yte, p_label):
    #######################################################################
    # Extract probability from p_val List
    P_val = []
    
    for i in range(0,len(p_val)):
        P_val.append(p_val[i][0])

#######################################################################
    
    sk_accuracy          = round(accuracy_score(yte, p_label),2)
    fpr, tpr, thresholds = metrics.roc_curve(yte, p_label, pos_label=1)
    sk_auc               = round(metrics.auc(fpr, tpr),2)
    sk_f1score           = round(f1_score(yte, p_label),2)
    sk_recall            = recall_score(yte, p_label, average=None)
    sk_precision         = round(sk_recall[0],2)
    sk_recallV           = round(sk_recall[1],2)
    sk_balacc            = round(balanced_accuracy_score(yte, p_label),2)
    sk_roc_auc           = roc_auc_score(yte, np.array(P_val))
    #######################################################################
    print("")
    print("==================================")
    print("**--** Metrics from sklearn **--**")
    print("==================================")
    print("")
    print("**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--***")
    print("**   {} Accuracy achieved from skleran metrics                         **".format(sk_accuracy))
    print("**   {} AUC recorded with sklearn metrics                               **".format(sk_auc))
    print("**   {0} recall value and {1} precision achieved with sklearn Metrics    **".format(sk_recallV,sk_precision))
    print("**   {} balanced accuracy obtained from sklearn Metric                  **".format(sk_balacc))
    print("**   {} F1 score achieved from sklearn Metric                           **".format(sk_f1score))
    print("**   {} ROC AUC score achieved with sklearn                             **".format(round(sk_roc_auc,2)))
    print("**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--***")
    sklearnMetrics = {"sk_accuracy" :sk_accuracy,
                      "sk_f1score"  :sk_f1score,
                      "sk_precision":sk_precision,
                      "sk_recallV"  :sk_recallV,
                      "sk_roc_auc"  :sk_roc_auc}
    
    return sklearnMetrics
    
    #######################################################################