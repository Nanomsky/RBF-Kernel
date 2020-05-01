# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:22:30 2020

@author: NN133
"""
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("C:/Users/NN133/Documents/libsvm-3.22/python")
from svmutil import *
#%matplotlib inline

from util_ker import *

#Import data
path = 'C:/Users/NN133/Documents/GitHub/GaussianKernelTest/data/breast-cancer-wisconsin.data.txt'
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

#Drop the id column
data.drop("id", axis=1, inplace=True)


#Extract Variables X and Label y from the data
X = data.iloc[:,:-1].values.reshape(data.shape[0],data.shape[1]-1)
y = data.iloc[:,-1].values.reshape(data.shape[0],1)

#SplitData into train, validation and Test data sets
xtr, xva, xte, ytr, yva, yte = splitdata(X, y, 25, 0.9)


#Choose Kernel
kernel = ['linear','H_poly','poly','rbf','erbf'] #['laplace','sqrexp','sigmoid']
ker


#Set Kernel parameter
params = {}
params['linear'] = []
params['H_poly'] = [2,3,4]
params['poly']   = [2,3,4]
params['rbf']    = [ 0.001,1.0,100.0]
params['erbf']   = [ 0.001,1.0,100.0]

#Set Kernel parameter
TrainKernel = {}
TestKernel  = {}
TrainKernelTime = {}
TestKernelTime = {}
PSDCheck    = {}
Perf_eva = {}
AucRoc = {}
Result = {}

#Construct Kernel
for ker in kernel:
    
    for par in range(len(params[ker])):
        
        k_param = params[ker][par]
        start_time=time.time()
        TrainKernel[ker] = kernelfun(xtr, xtr, ker, k_param)
        end_time=time.time()
        TrainKernelTime[ker] = end_time - start_time
        print('{} minutes to construct Training kernel'.format(ker_time/60))
        PSDCheck[ker]   = checkPSD(TrainKernel[ker])
        plt.imshow(TrainKernel[ker]) #Any other kernel analysis can be inserted here
        TrainKernel[ker] = np.multiply(np.matmul(ytr,ytr.T),TrainKernel[ker])
        TrainKernel[ker] = addIndexToKernel(TrainKernel[ker])
        
        start_time=time.time()
        TestKernel[ker] = kernelfun(xtr, xte, ker, k_param)
        end_time=time.time()
        TestKernelTime[ker] = end_time - start_time
        print('{} minutes to construct Test kernel'.format(ker_time/60))
        TestKernel[ker] = addIndexToKernel(TestKernel[ker])
   
        model = svm_train(list(ytr), [list(r) for r in TrainKernel[ker]], ('-b 1 -c 4 -t 4'))
        p_label, p_acc, p_val = svm_predict(list(yte),[list(row) for row in TestKernel[ker]], model, ('-b 1'))
        Perf_eva[ker] = EvaluateTest(np.asarray(yte/1.),np.asarray(p_label))
        print("--> {} F1 Score achieved".format(Evaluation["Fscore"]))
        AucRoc[ker] = computeRoc(yte, p_val)
        Result[ker+'_'+ str(par)] = (TrainKernel,TrainKernelTime,PSDCheck,
               TestKernel,TestKernelTime,model,p_label, p_acc, p_val,Perf_eva,AucRoc)
        
        
    print('-' * 6)
    print(' Done ')
    print('=' * 6)



 print("K_tr_" + ker)

#initialize the kernel matrix
K_tr,K_te = intitializeKernels(m,n)

#Append an index column to the kernel matrix
H2 = addIndexToKernel(K_te)

RecordTime = {}

x=X[:10,:]
#Choose Parameter
params=[ 0.001, 0.01, 0.1, 1.0, 10.0, 100.0 ]





#Use Single Kernel
#Kernel = ['rbf']
#ker = Kernel[0]



#####
start_time2 = time.time()
H1 = kernelfun(xtr,xte, ker, params)
end_time2 = time.time()

####

for i in range(0,n):
    for j in range(0,m):
        u = K_tr[i,:]
        print(u)
        v = K_tr[j,:]
        print(v)
        K_tr[i,j] = np.exp(-(np.dot((u-v),(u-v).T)/2 * (1.25**2)))
        
        
#Check if Kernel is PSD      
checkPSD(K_tr)

#plot kernel with plt.imshow()
plt.imshow(K_tr)

#Multiply kernel by label
K_tr = np.multiply(np.matmul(ytr,ytr.T),K_tr)

#Append index column to the kernel matrix 
K_tr = addIndexToKernel(K_tr)

#Evaluation = EvaluateTest(np.asarray(p_label),yte)
Evaluation = EvaluateTest(np.asarray(yte/1.),np.asarray(p_label))
print("--> {} F1 Score achieved".format(Evaluation["Fscore"]))