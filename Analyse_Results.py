# -*- coding: utf-8 -*-
"""
Created on Thu May  7 03:28:46 2020

@author: NN133
"""
def Analyse_Results(Exp):
    '''
    This analyses and extracts data from the output of the fitkernelmodel
    function. It outputs two DataFrames showing comparative results obtained
    from testing one or more kernels
    '''
    #################################################################
    #Initialise variables
    ker_list, col_names = [],[]
    Stage1, Stage2, Stage3, Stage4 =[],[],[],[]
    train_kernel_time, test_kernel_time=[],[]
    diag_Dominance,is_PSD = [],[]
    num_Of_SV,AUC,Eva  = [],[],[]
    
    #################################################################
    
    Key1 = list(Exp.keys()) #Extract keys from the input dictionary into a list 
    
    for i in Key1:
        ker_list.append(list(Exp[str(i)].keys())) #Extract list of kernels
        for k in Exp[str(i)].keys():
            col_names.append(k) #Extract kernel names  
            Stage1.append(Exp[str(i)][k][0]) #Extract data created during the 4 stages
            Stage2.append(Exp[str(i)][k][1]) 
            Stage3.append(Exp[str(i)][k][2])
            Stage4.append(Exp[str(i)][k][3])
            
    n = len(col_names) # also number of kernels to analyse
    
    #################################################################
    #Visual plot of the kernels
    fig = plt.figure(figsize=(15,20))        
    for i in range(len(col_names)):
        plt.subplot(np.ceil(n/3),3,i+1)
        plt.imshow(Stage1[i][0])
        plt.xlabel(str(col_names[i]), fontsize=15)
    plt.tight_layout()
    plt.show()
    
    ##################################################################
    #Extract the remaining variables and append to the relevant list
    for i in range(n):
        Eva.append(list(Stage4[i][0].values()))
        train_kernel_time.append(Stage1[i][1])
        test_kernel_time.append(Stage2[i][1])
        diag_Dominance.append(Stage1[i][2])
        is_PSD.append(Stage1[i][3])
        num_Of_SV.append(Stage3[i][0])
        AUC.append(Stage4[i][1])  
    
    ##################################################################
    #Store the extracted data in DataFrames
    perfEva = pd.DataFrame(Eva)
    perfEva.columns = [key for key in Stage4[0][0].keys()]
    perfEva.index = col_names 
    
    print(perfEva)
    Kernel_Analysis = [train_kernel_time, test_kernel_time,
                      diag_Dominance, is_PSD, num_Of_SV,AUC]
    
    Kernel_index    = ['train_kernel_time','test_kernel_time',
                      'diag_Dominance','is_PSD','num_Of_SV','AUC']
    
    Kernel_Analysis_df = pd.DataFrame(Kernel_Analysis)
    Kernel_Analysis_df =Kernel_Analysis_df.T
    Kernel_Analysis_df.columns = Kernel_index
    Kernel_Analysis_df.index= col_names 
    
    print(Kernel_Analysis_df)
    return perfEva, Kernel_Analysis_df
