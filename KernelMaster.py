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
from libsvm.svmutil import * #for desktop

import seaborn as sns
sns.set()
#sys.path.append("C:/Users/NN133/Documents/libsvm-3.22/python") #for laptop
#from svmutil import *
from util_ker import *
from util_sk import *
#Import data

path = '/Users/osita/Documents/data/wdbc/breast-cancer-wisconsin.data.txt'
#path = 'C:/Users/NN133/Documents/GitHub/GaussianKernelTest/data/breast-cancer-wisconsin.data.txt'
col_names = ['id','Clump_Thick','U_Cell_Size', 'U_Cell_Shape','Marg_Adh','Epith_Cell_Size','Bare_Nuclei',
            'Bland_Chrom','Norm_Nucle','Mitoses','Class']

df = pd.read_csv(path,header=None, names = col_names)
#df.info() #Check the data types

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

# Plotting with Seaborn
varN = list(data)
g = sns.PairGrid(data, vars=varN, hue="Class", palette="Set1",

                 hue_kws={"marker": ["o", "s"]})


g.map_diag(plt.hist)
#g.map(plt.scatter, alpha=0.8)
g.map_offdiag(plt.scatter);
g.add_legend();

#############################################################################
#Other plat from the Kaggle heart disease classification ML 
#data.Class.value_counts()
sns.countplot(x="Class", data=data, palette = "bwr")
plt.show()

#############################################################################

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

fig = plt.figure()
data.Class.value_counts()
plt.show()
##############################################################################
fig = plt.figure()
sns.heatmap(data.iloc[:,:-1].corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(13,11)
plt.show()

#############################################################################

#Extract Variables X and Label y from the data
X = data.iloc[:,:-1].values.reshape(data.shape[0],data.shape[1]-1)
y = data.iloc[:,-1].values.reshape(data.shape[0],1)

#SplitData into train, validation and Test data sets
xtr, xva, xte, ytr, yva, yte = splitdata(X, y, 12, 0.8)

#################
#Sample data 
#xtr = xtr[:10,:]
#ytr = ytr[:10]

#xte = xte[:4,:]
#yte = yte[:4]
################

#Choose Kernel
#kernel = ['linear','H_poly','poly','rbf','erbf'] #['laplace','sqrexp','sigmoid']
kernel = ['linear','poly','H_poly','rbf']
#kernel = ['laplace']
        
#Set Kernel parameter
params = {}
params['linear']  = [1]
params['H_poly']  = [2,3,4]
params['poly']    = [2,3,4]
params['rbf']     = [0.001,0.01,0.1,1,10,100,1000]
params['erbf']    = [0.001,1.0,100.0]
params['laplace'] = [0.00001,0.0001,0.001,0.01]


Exp = fitkernelmodel(xtr, xte, ytr, yte, kernel, params)

Perf, Analysisdf = Analyse_Results(Exp)

plotResult(Perf)
#############################################################################
