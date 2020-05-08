#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:17:57 2020

@author: osita
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  1 02:40:10 2020

@author: NN133
"""
#%reset
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *
#sys.path.append("C:/Users/NN133/Documents/libsvm-3.22/python")
#from svmutil import *
from util_ker import *

#Import data

path = '/Users/osita/Documents/data/wdbc/breast-cancer-wisconsin.data.txt'
#path = 'C:/Users/NN133/Documents/GitHub/GaussianKernelTest/data/breast-cancer-wisconsin.data.txt'
col_names = ['id','Clump_Thick','U_Cell_Size', 'U_Cell_Shape','Marg_Adh','Epith_Cell_Size','Bare_Nuclei',
            'Bland_Chrom','Norm_Nucle','Mitoses','Class']

df = pd.read_csv(path,header=None, names = col_names)
df.info() #Check the data types

#Extract the index for Bare_Neclei values '?'
ind = df.query("Bare_Nuclei=='?'").index

#drop the rows with values '?' 
data = df.drop(ind, axis ='index')

#Convert the Bare_Nuclei datatype from Object to int64
data['Bare_Nuclei'] = data.Bare_Nuclei.astype('int64')

#Drop the id column
data.drop("id", axis=1, inplace=True)

#Check for null values
data.isnull().sum()

#Look up Summary statistics of the data
Summary_Stats = data.iloc[:,:-1].describe()

#plot the mean values from the summary stats bar
fig = plt.figure(figsize=(6,6))
Summary_Stats.loc['mean',:].plot(kind='barh', xerr=Summary_Stats.loc['std',:]);
plt.title('Bar chart showing the mean and std of variables')
plt.xlabel('Mean')

#plot the mean values from the summary stats line
fig = plt.figure(figsize=(9,4))
Summary_Stats.loc['mean',:].plot(kind='line', color='blue', linewidth=3);
Summary_Stats.loc['std',:].plot(kind='line', color='lightgreen', linewidth=2)
plt.legend

#Plot the class distribution
fig = plt.figure(figsize=(15,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.bar(['neg','pos'], data.Class.value_counts().values, color=('grey','maroon'))
ax1.legend(['neg','pos'])
ax1.set_xlabel('Class Labels')
ax1.set_ylabel('Examples')
Explode=[0,0.2] #Separates the section of the pie chart specified
ax2.pie(data.Class.value_counts().values,explode=Explode, shadow=True,startangle=45)
ax2.legend(['neg','pos'],title ="Classes")

#Replace class labels from [benign, malignant]=(2,4) to (-1,1)
data.Class.replace({2:-1,4:1}, inplace=True) 
data.Class.value_counts()


#Extract Variables X and Label y from the data
X = data.iloc[:,:-1].values.reshape(data.shape[0],data.shape[1]-1)
y = data.iloc[:,-1].values.reshape(data.shape[0],1)

#SplitData into train, validation and Test data sets
xtr, xva, xte, ytr, yva, yte = splitdata(X, y, 25, 0.8)

#################
#Sample data 
#xtr = xtr[:10,:]
#ytr = ytr[:10]

#xte = xte[:4,:]
#yte = yte[:4]
################

#Choose Kernel
#kernel = ['linear','H_poly','poly','rbf','erbf'] #['laplace','sqrexp','sigmoid']
kernel = ['linear','poly','rbf']

#Set Kernel parameter
params = {}
params['linear'] = [1]
params['H_poly'] = [2,3,4]
params['poly']   = [2,3,4,5]
params['rbf']    = [100.0,0.001,1.0]
params['erbf']   = [ 0.001,1.0,100.0]


ANS = []

#Construct Kernel
for ker in kernel:
#Initialize Dictionaries
    TrainKernel = {}
    TestKernel  = {}
    TrainKernelTime = {}
    TestKernelTime = {}
    PSDCheck    = {}
    Perf_eva = {}
    AucRoc = {}
    Result = {}
    Result1 =()
    

    for par in range(len(params[ker])):
        k_param = params[ker][par]
        start_time=time.time()
        TrainKernel[ker] = kernelfun(xtr, xtr, ker, k_param)
        end_time=time.time()
        TrainKernelTime[ker] = end_time - start_time
        print('{} minutes to construct Training kernel'.format(TrainKernelTime[ker]/60))
        print('')
        
        Result1 = tuple(TrainKernel)
        PSDCheck[ker]   = checkPSD(TrainKernel[ker])
        #plt.imshow(TrainKernel[ker]) #Any other kernel analysis can be inserted here
        TrainKernel[ker] = np.multiply(np.matmul(ytr,ytr.T),TrainKernel[ker])
        TrainKernel[ker] = addIndexToKernel(TrainKernel[ker])
        print('=========>'+ ker + '-'+ str(par)+'=========>')
        
        start_time=time.time()
        TestKernel[ker] = kernelfun(xtr, xte, ker, k_param)
        end_time=time.time()
        TestKernelTime[ker] = end_time - start_time
        print('\n')
        print('{} minutes to construct Test kernel'.format(TestKernel[ker]/60))
        TestKernel[ker] = addIndexToKernel(TestKernel[ker])
        print('=========>'+ ker + '-'+ str(par)+'=========>')
        
        model = svm_train(list(ytr), [list(r) for r in TrainKernel[ker]], ('-b 1 -c 4 -t 4'))
        p_label, p_acc, p_val = svm_predict(list(yte),[list(row) for row in TestKernel[ker]], model, ('-b 1'))
        Perf_eva[ker] = EvaluateTest(np.asarray(yte/1.),np.asarray(p_label))
        AucRoc[ker] = computeRoc(yte, p_val)
        AddTrainKer = Result1 + (TrainKernelTime,PSDCheck,
               TestKernel,TestKernelTime,model,Perf_eva,AucRoc)
        Result[ker +'_'+ str(par)] =  AddTrainKer
        print('=========>'+ ker + '-'+ str(par)+'=========>')
        del p_val
    ANS.append(Result)
        
print('-' * 6)
print(' Done ')
print('=' * 6)

'''
ANS = []
Out = []
#Construct Kernel
for ker in kernel:
#Initialize Dictionaries
    TrainKernel, TestKernel= {},{}
    TrainKernelTime, TestKernelTime = {},{}
    PSDCheck, Perf_eva, AucRoc = {},{},{}
    Result = {}
    Result1 =()
    NanoResult = {}
   
    for par in range(len(params[ker])):
        Nano_time1 = []
        Nano_time2 = []
        Nano_ker1 = []
        Nano_ker2 = []
        Nano_PSD  = []
        Nano_Model = []
        Nano_Perf = []
        Nano_AUC = []
        Nano_diag = []
        ################################################
        k_param = params[ker][par]#Calls the param dict by the ker key and slices therough params with iterable variable par
        start_time=time.time() #Check points the start time
        TrainKernel[ker] = kernelfun(xtr, xtr, ker, k_param) # Kernel added to a dict. Key refers to selected kernel
        # What is the point since the key will not change till the completed iterations through params
        
        Nano_ker1.append(TrainKernel[ker]) #This would have been an alternative solution but this 
        Nano_diag.append(diag_dominace(TrainKernel[ker]))
        
        end_time=time.time()
        TrainKernelTime[ker] = end_time - start_time
        Nano_time1.append(TrainKernelTime[ker]) 
        
        print('{} minutes to construct Training kernel'.format(TrainKernelTime[ker]/60))
        print('')
        
        Result1 = tuple(TrainKernel)
        PSDCheck[ker]   = checkPSD(TrainKernel[ker])
        Nano_PSD.append(PSDCheck[ker]) 
        
        fig = plt.figure()
        plt.imshow(TrainKernel[ker]) #Any other kernel analysis can be inserted here
        plt.show()
        TrainKernel[ker] = np.multiply(np.matmul(ytr,ytr.T),TrainKernel[ker])
        TrainKernel[ker] = addIndexToKernel(TrainKernel[ker])
        print('\n')
        print('=========>'+ ker + '-'+ str(par)+'=========>')
        
        start_time=time.time()
        TestKernel[ker] = kernelfun(xtr, xte, ker, k_param)
        end_time=time.time()
        TestKernelTime[ker] = end_time - start_time
        Nano_time2.append(TestKernelTime[ker])
        
        print('\n')
        print('{} minutes to construct Test kernel'.format(TestKernel[ker]/60))
        TestKernel[ker] = addIndexToKernel(TestKernel[ker])
        print('=========>'+ ker + '-'+ str(par)+'=========>')
        print('\n')
        model = svm_train(list(ytr), [list(r) for r in TrainKernel[ker]], ('-s 0 -b 1 -c 10 -t 4'))
        p_label, p_acc, p_val = svm_predict(list(yte),[list(row) for row in TestKernel[ker]], model, ('-b 1'))
        Nano_Model.append(model)
        
        Perf_eva[ker] = EvaluateTest(np.asarray(yte/1.),np.asarray(p_label))
        Nano_Perf.append(Perf_eva[ker])
        
        AucRoc[ker] = computeRoc(p_label, p_val)
        Nano_AUC.append(AucRoc[ker])
        
        AddTrainKer = Result1 + (TrainKernelTime,PSDCheck,
               TestKernel,TestKernelTime,model,Perf_eva,AucRoc)
        Result[ker +'_'+ str(par)] =  AddTrainKer
        
        print('=========>'+ ker + '-'+ str(par)+'=========>')
        sklearnMetrics = compMetrics(p_val, yte, p_label)
       
        if sklearnMetrics["sk_auc"] == AucRoc[ker]:
            print("AUC from sklearn the same as AUC computed here")
        else:
            print('Something is Wrong')
        
        NanoResult[ker +'_'+ str(par)] = (Nano_time1, 
          Nano_time2,
          Nano_ker1, 
          Nano_ker2, 
          Nano_PSD, 
          Nano_Model, 
          Nano_Perf,
          Nano_AUC, 
          Nano_diag
          ) 
    ANS.append(Result)
    Out.append(NanoResult)  

    
print('-' * 6)
print(' Done ')
print('=' * 6)

########################################
#Extract Kernel performance
     
train_k_time = []
test_k_time = []
is_PSD = []
col_names = []
AUC = []
Eva = []
A=[]
diagN=[]
num = len(Out)
for i in range(num):
    for k in Out[i]:
        col_names.append(k)
        train_k_time.append(Out[i][k][0][0])
        test_k_time.append(Out[i][k][1][0])
        is_PSD.append(Out[i][k][4][0])
        Eva.append(list(Out[i][k][6][0].values()))
        A.append(Out[i][k][6][0])
        AUC.append(Out[i][k][7][0]) 
        diagN.append(Out[i][k][8][0]) 

perfEva = pd.DataFrame(Eva)
perfEva_T = perfEva.T
perfEva_T.columns = col_names
perfEva_T.index = [keys for keys in A[0].keys()]
print('\n')
print('DataFrame showing prediction performance of kernels')
print('\n')
print(perfEva_T)

Tab_index = 'train_k_time','test_k_time' ,'is_PSD','AUC','diagN'
Tab = [train_k_time,test_k_time ,is_PSD,AUC,diagN ]       
Tab_df = pd.DataFrame(Tab)
Tab_df.index = Tab_index
Tab_df.columns = col_names
print('\n')
print('Other kernel performance indicators measured' )
print('\n')
print(Tab_df)

perfEva_T.loc['Pos',:].plot(kind='bar')
'''