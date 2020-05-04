#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:24:19 2020

@author: osita
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



###############################################################################
# edit distance

def editdist(X,Y):
    '''
    Computes edit distance between two strings 
    
    Input
    =====
    s, t = Two strings 
    
    Output
    ======
    Outputs an integer value indicating the distance between the two strings
    '''
    
    m = len(X)
    n = len(Y)
    D = np.zeros((m+1,n+1))
    
    for i in range(0,m+1):
        D[i,0] = i
    
    for j in range(0,n+1):
        D[0,j] = j
    
    for i in range(1,m+1):
        for j in range(1,n+1):
            
            if X[i-1] == Y[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + 2) #deletion, insertion, substitution
             
    return D[-1,-1]

###############################################################################
# edit distance

def editdist_norm(X,Y):
    '''
    Computes edit distance between two strings and normalises this by the number
    of items common to both strings being compared
    
    Input
    =====
    s, t = Two strings 
    
    Output
    ======
    Outputs an integer value indicating the distance between the two strings
    '''
    
    m = len(X)
    n = len(Y)
    D = np.zeros((m+1,n+1))
    
    for i in range(0,m+1):
        D[i,0] = i
    
    for j in range(0,n+1):
        D[0,j] = j
    
    for i in range(1,m+1):
        for j in range(1,n+1):
            
            if X[i-1] == Y[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + 2) #deletion, insertion, substitution
    
    dist = D[-1,-1]
    num_intersect = len(list(set(X).intersection(set(Y))))
    
    if num_intersect == 0:
        edit_dist = dist/1
    else:
        edit_dist = dist/num_intersect
    
    return edit_dist

###############################################################################
# edit distance
    
def editdist_norm_max(X,Y):
    '''
    Computes edit distance between two strings and normalises with the maximum lenth of
    both strings
    
    Input
    =====
    s, t = Two strings 
    
    Output
    ======
    Outputs an integer value indicating the distance between the two strings
    '''
    
    m = len(X)
    n = len(Y)
    D = np.zeros((m+1,n+1))
    
    for i in range(0,m+1):
        D[i,0] = i
    
    for j in range(0,n+1):
        D[0,j] = j
    
    for i in range(1,m+1):
        for j in range(1,n+1):
            
            if X[i-1] == Y[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + 2) #deletion, insertion, substitution
    
    dist = D[-1,-1]
    
    maxlength = max(m,n)    
    if maxlength == 0:
        edit_dist = dist/1
    else:
        edit_dist = dist/maxlength
    
    return edit_dist

###############################################################################  
# edit distance
    
def editdist_norm_intersect(X,Y):
    '''
    Computes edit distance between two strings and normalises with the maximum lenth of
    both strings
    
    Input
    =====
    s, t = Two strings 
    
    Output
    ======
    Outputs an integer value indicating the distance between the two strings
    '''
    
    m = len(X)
    n = len(Y)
    D = np.zeros((m+1,n+1))
    
    for i in range(0,m+1):
        D[i,0] = i
    
    for j in range(0,n+1):
        D[0,j] = j
    
    for i in range(1,m+1):
        for j in range(1,n+1):
            
            if X[i-1] == Y[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + 2) #deletion, insertion, substitution
    
    dist = D[-1,-1]
    num_intersect = len(list(set(X).intersection(set(Y))))
    edit_dist = dist/(2**num_intersect)
    
    return edit_dist 

###############################################################################
    # edit distance

def editdist_Levenshtein(X,Y):
    '''
    Computes Levenshtein edit distance between two strings 
    
    Input
    =====
    s, t = Two strings 
    
    Output
    ======
    Outputs an integer value indicating the distance between the two strings
    '''
    
    m = len(X)
    n = len(Y)
    D = np.zeros((m+1,n+1))
    
    for i in range(0,m+1):
        D[i,0] = i
    
    for j in range(0,n+1):
        D[0,j] = j
    
    for i in range(1,m+1):
        for j in range(1,n+1):
            
            if X[i-1] == Y[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + 1) #deletion, insertion, substitution
             
    return D[-1,-1]

###############################################################################


# Simple Kernels
"""
Created on Sun Jan 26 03:03:06 2020
    1) linear
    2) poly
    3) rbf
    4) erbf
    5) laplace
    6) sqrexp -Squared Exponential 
@author: NN133

"""
def kernelfun(X,Y, kernel, params):
    
    m = X.shape[0] #,X.shape[1]
    n = Y.shape[0] #,Y.shape[1]
    H = np.zeros((n,m))
    
    if kernel == 'linear':
        H = np.dot(Y,X.T)
    
    elif kernel == 'H_poly': 
        #Homogeneous polynomial kernel. All monomials of degree d
        #H = np.dot(Y,X.T)
        H = np.dot(Y,X.T)**params
    elif kernel == 'poly': 
        #Non - Homogeneous polynomial kernel
        H = (np.dot(Y,X.T) + 1) ** params
    
    elif kernel =='rbf':
         for i in range(0,n):
            for j in range(0,m):
                u = Y[i,:]
                v = X[j,:]
                H[i,j] = np.exp(-(np.dot((u-v),(u-v).T)/2 * (params**2)))

    elif kernel =='erbf':
         for i in range(0,n):
            for j in range(0,m):
                u = Y[i,:]
                v = X[j,:]
                H[i,j] = np.exp(-np.sqrt(np.dot((u-v),(u-v).T))/(2*(params**2)))

    elif kernel =='laplace':
         for i in range(0,n):
            for j in range(0,m):
                u = Y[i,:]
                v = X[j,:]
                H[i,j] = np.sum(np.exp(-(np.abs(u-v)/params[0])))

    elif kernel =='sqrexp': #Squared exponential kernel
         for i in range(0,n):
            for j in range(0,m):
                u = X[i,:]
                v = X[j,:]
                H[i,j] = (params[0] * np.exp(-0.5* params[1]*np.dot((u-v),(u-v).T)))
    
    #elif kernel =='sigmoid': #Squared exponential kernel
    
    # edit distance kernels 
    elif kernel == 'editdist':
        for i in range(0,n):
            for j in range(0,m):
                u = Y[i,:][0]
                v = X[j,:][0]
                H[i,j] = editdist(u,v) 
    
    elif kernel == 'editdist_Levenshtein':
        for i in range(0,n):
            for j in range(0,m):
                u = Y[i,:][0]
                v = X[j,:][0]
                H[i,j] = editdist_Levenshtein(u,v) 
        
    elif kernel == 'editdist_norm_intersect':
        for i in range(0,n):
            for j in range(0,m):
                u = Y[i,:][0]
                v = X[j,:][0]
                H[i,j] = editdist_norm_intersect(u,v)
        
    elif kernel == 'editdist_norm_max':
        for i in range(0,n):
            for j in range(0,m):
                u = Y[i,:][0]
                v = X[j,:][0]
                H[i,j] = editdist_norm_max(u,v)
        
    elif kernel == 'editdist_norm':
        for i in range(0,n):
            for j in range(0,m):
                u = Y[i,:][0]
                v = X[j,:][0]
                H[i,j] = editdist_norm(u,v)     
        
    return H

###############################################################################
#Test for positive definiteness
def checkPSD(K):
    if np.all(np.linalg.eigvals(K)):
        
        print('--> Kernel is valid and PSD')
    else:
        print('--> Kernel is not PSD')
        
    return np.all(np.linalg.eigvals(K))

###############################################################################
    """
Created on Fri Jan 24 21:45:57 2020
The kernel Alignment function tests the similarity of two kernels. K = 1
indicates the matrices are equal. While K = 0 means they are orthogonal and 
therefore dissimilar. The kernel alignment therefore is an indication of how
close/similar two kernel matrices are. The value K > 0 means K1 is psd for every K2 
that is psd. The alignment can be viewed as the cosine of the angle between the 
matrices
@author: NN133
"""

def kernelAlignment(K1,K2):
    '''
    Kernel alignment to test the similarity of two kernels
    
    Input
    =====
    K1, K2 = Two kernel to test their alignment
    
    Output
    ======
    K = Value indicating a measure of similarity between kernel 1 and 2
    '''
       
    k1 = np.sum(np.dot(K1,K2)) 
    k2 = np.sum(np.dot(K1,K1))  
    k3 = np.sum(np.dot(K2,K2))
    k4 = np.sqrt(np.dot(k2,k3))
    K  = np.divide(k1, k4)
    
    if K == 1:
        print('Both kernels are equal')
    if K == 0:
        print('Both kernels are orthogonal')
    if K > 0:
        print('A positive value is an indication kernel 1 is psd for every kernel 2 that is psd')
    
    return K

###############################################################################
# Evaluation
    """
Created on Fri Jan 31 06:52:41 2020

@author: NN133
"""

def EvaluateTest(ylabel, Pred):
    
    Evaluation = {}
    FN,FP,TP,TN = 0,0,0,0
   
    for i in range(0,ylabel.shape[0]):
        if (ylabel[i]==Pred[i]).any():
            if (Pred[i]==1).any():
                TP+=1
            elif Pred[i]!=1:
                TN+=1
        if (ylabel[i]!=Pred[i]).any():
            if (Pred[i]==1).any():
                FP+=1
            elif Pred[i]!=1:
                FN+=1
    TOTAL = TP + TN + FP + FN
    TPN   = TP+ TN
    
    print("--> The total of {0} predicted with only {1} accurate predictions".format(TOTAL,TPN))
    print('')
    print('='*25)
    print('Ground Truth comparison')
    print('='*25)
    print("Actual label is True while we predicted True - True Positive",format(TP))
    print("Actual label is False while we predicted True - False Positive",format(FP))
    print("Actual label is True while we predicted False - False Negative",format(FN))
    print("Actual label is False while we predicted False - True Negatve",format(TN))  
    print('') 
    #try:
    Pos        = TP+FP                                   # sum of TP and FP
    Neg        = TN+FN
    accu       = np.round(((TP+TN)/(TP+FN+FP+TN)*100),2)
    #sen        = TP/(TP+FN)  
    if (TP+FN) == 0:
        sen    = 0
        miss   = 0
        recall = 0
        print("No True positives or False negatives predicted")
        print("Sensitivity set to zero 0")
        print("Miss (false negative rate) set to 0")
        print("Recall value set to 0")
        print('='*45)
    else:
        sen    = np.round(TP/(TP+FN),2)      # true positive rate,sensitivity,recall
        miss   = np.round(FN/(TP+FN),2)      # false negative rate, miss
        recall = np.round(TP/(TP+FN),2)      # Recall describes the completeness of the classification
   
    if (TN + FP) == 0:
        spec = 0
        fall = 0
        print("No True positives or False negatves predicted")
        print("Specificity set to 0")
        print("Fallout (false positive rate) set to 0")
        print('='*45)
    else:
        spec   = np.round(TN/(TN+FP),2)     # true negative rate, specificity7
        fall   = np.round(FP/(TN+FP),2)     # false positive rate, fallout
    
    if (TN+FN) == 0:
        NPV = 0
        print("No Negative outcomes predicted")
        print("Negative predicted value set to 0")
        print('='*45)
    else:
        NPV        = np.round(TN/(TN+FN),2)                  # negative predictive value
     
    if (TP+FP) == 0:
        precision = 0
        print("No True positives or False positives predicted")
        print('='*45)
    else:
        precision  = np.round(TP/(TP+FP),2)                  # precision measures the actual accuracy of the classification
        
    RPP        = np.round((TP+FP)/(TP+FN+FP+TN),2)           # rate of positive predictions
    RNP        = np.round((TN+FN)/(TP+FN+FP+TN),2)           # rate of negative predictions
    
    if (precision + recall) == 0:
        Fscore = 0
        print("Fscore cannot be calculated as denominator is 0")
        print('='*45)
    else:
        Fscore = np.round(2 * ((precision * recall) / (precision + recall)),2)
    
    
    print("--> {} positive outcomes prediticted".format(Pos))
    print("--> {} negative outcomes prediticted".format(Neg))
    print("--> An accuracy of {} % was achieved".format(accu))
    print("--> Sensitity of {} was achieved".format(sen))
    print("--> Specificity of {} was achieved ".format(spec))
    print("--> {} rate of positive prediction".format(RPP))
    print("--> {} rate of negative prediction".format(RNP))
    print("--> {} false negative rate was achieved".format(miss))
    print("--> {} false positve rate (fallout) was achieved".format(fall))
    print("--> Negative predictive value of {}".format(NPV))
    print("--> Recall value 0f {} achieved".format(recall))
    print("--> The precision vaue of {} achieved".format(precision))
    print("--> An Fscore of {} achieved".format(Fscore))
    
    
    confusion_mat = np.array([[TN, FP], [FN, TP]])
    
    
    plt.figure(figsize=(8,4))

    #plt.suptitle("Confusion Matrixes",fontsize=24)
    plt.title("Confusion Matrix")
    plt.subplots_adjust(wspace = 0.1, hspace= 0.01)
    sns.heatmap(confusion_mat,annot=True,cmap="Reds",fmt='.2g',cbar=True, annot_kws={"size":15})
    plt.show()
    
    Evaluation = {"Pos": Pos, "Neg": Neg, "accu": accu,"sen": sen,
                  "spec": spec, "RPP": RPP, "RNP": RNP, "miss": miss,
                  "fall":fall, "NPV": NPV,"recall":recall, "precision":precision,
                  "Fscore":Fscore}
    
    return Evaluation

###############################################################################
# Normalise a Kernel    
"""
Created on Tue Jan 21 22:48:39 2020

@author: NN133

For more info, see www.kernel-methods.net

"""

def normalise(K):
    '''
    Normalizes kernel K
    
    Input
    =====
    K =  un-normalized kernel K
    
    Output
    ======
    Kc =  Normalized kernel
    '''

    D  = np.diag(1/np.sqrt(np.diag(K)))
    Kc = np.matmul(D,np.matmul(K, D))

    return Kc 
    
###############################################################################
# Compute ROC
"""
Created on Sat Feb  1 03:25:51 2020

@author: NN133
"""

def computeRoc(p_label, p_val):
    '''
    Computes Receiver Operating Characteristics (ROC) Area Under Curve (AUC)
    
    Input
    =====
    p_label =   predicted labels
    p_val   =   probability values for the predicted labels
    
    Output
    ======
    AUC value
    '''
    A =  p_val.copy()
    
    l2 = [p_val.index(x) for x in sorted(p_val,reverse=True) ]
    c=[]
    Y=[]
    for i in l2:
        
        c.append(A[i])
        Y.append(p_label[i])
    
    Ya =np.asarray(Y)
    stack_x = np.cumsum(Ya == -1)/np.sum(Ya ==-1) 
    stack_y = np.cumsum(Ya == 1)/np.sum(Ya == 1) 
    L = len(Ya)
    
    auc = np.sum(np.multiply((stack_x[1:L]-stack_x[0:L-1]),(stack_y[1:L])))
    
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(stack_x,stack_y)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve of AUC = {} ".format(round(auc, 2)))
    plt.show()
    
    print("--> An AUC value of {} achieved".format(round(auc,2)))
    
    return auc

###############################################################################
# Split Data
    
def splitdata(X, Y, rand_seed, tnx):
    '''
    Function used to split data into training, test and validation datastes
    This takes the predictor variables X and response variables Y, and 
    
    Input
    =====
        X         = An m by nx (nx = number of features) data matrix
        Y         = An m by 1 array of class labels
        rand_seed = Integer to ensure reproducibility for random generation 
        tnx       = Float between 0 and 1 used to specify the size of test/validation
    
    Output
    ======
        xtr, ytr = Training data, label
        xva, yva = Validation data, label
        xte, yte = Test data, label
    '''
    np.random.seed(rand_seed)
    m    = X.shape[0]
    index = np.random.permutation(m)
    
    if (tnx > 1) or (tnx < 0) :
        print("This should be greater than 0 and less than 1")

    len1= int(np.round(len(index)* tnx, 0))
    len2= int(np.round(len(index)* (1-tnx)/2, 0))

    xtr  = X[index[0:len1],:]
    xva  = X[index[len1:(len1 + len2)],:]
    xte  = X[index[(len1 + len2):],:]
    
    ytr  = Y[index[0:len1]]
    yva  = Y[index[len1:(len1 + len2)]]
    yte  = Y[index[(len1 + len2):]]
    
    print('{} training examples and {} features'.format(xtr.shape[0],xtr.shape[1]))
    print('{} validation examples and {} features'.format(xva.shape[0],xva.shape[1]))
    print('{} testing examples and {} features'.format(xte.shape[0],xte.shape[1]))
    
    return xtr, xva, xte, ytr.reshape(len(ytr),1), yva.reshape(len(yva),1), yte.reshape(len(yte),1)

###############################################################################
def intitializeKernels(m,n):
    
    '''
    This function initializes the Training and Test Kernel Matrices
    m = number of training examples
    n = number of test examples 
    '''
    Train_kernel = np.zeros((m,m))
    Test_kernel  = np.zeros((n,m))
    
    return Train_kernel,Test_kernel

###############################################################################
def addIndexToKernel(K_mat):
    '''
    This function appends an index column to the contructed kernel matrix
    '''
    n = K_mat.shape[0]
    Hess_mat = np.concatenate((np.arange(n)[:,np.newaxis]+1, K_mat),axis=1)
    
    return Hess_mat