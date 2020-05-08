# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:50:22 2020

@author: NN133
"""
def fitkernelmodel(xtr, xte, ytr, yte, kernel, params):
    
    Out = {}
    for ker in kernel:
        Result={}
        for par in range(len(params[ker])):
            
            k_param = params[ker][par] #Select the parameter k_param for specified kernel 
            #################################################################################
            
            start_time=time.time() #Check point to start timer
            K_mat = kernelfun(xtr, xtr, ker, k_param) #Construct Kernel
            end_time=time.time() #Check points the end time
            Kernel_Time = end_time - start_time #Computes the time taken to construct the kernel
            dom = diag_dominace(K_mat) #Test for diagomal dominance
            PSDCheck = checkPSD(K_mat) #Checks if kernel is PSD
            
            Stage1 = (K_mat,Kernel_Time,dom,PSDCheck) #Create a tuple of variables to save
            
            K_mat = numericStability(K_mat) #Check for numerical stability
            Trainkernel= np.multiply(np.matmul(ytr,ytr.T),K_mat) #Contrusts the Hessian Matrix
            Hessian = addIndexToKernel(Trainkernel) #Appends an index to the Hessian
            
            #################################################################################
            
            start_time=time.time()#Check point to start timer
            Test_K_mat = kernelfun(xtr, xte, ker, k_param) #Construct test kernel
            end_time=time.time() #Check point to end timer
            Test_Kernel_time = end_time - start_time #Computes the time taken to construct the kernel
            
            Stage2 = (Test_K_mat,Test_Kernel_time) #Create a tuple of variables to save
            
            Test_K_mat = addIndexToKernel(Test_K_mat) #Add index to the constructed test kernel
            
            #################################################################################
            
            model = svm_train(list(ytr), [list(r) for r in Hessian], ('-s 0 -b 1 -c 10 -t 4'))
            numofSV = model.get_nr_sv()
            svIndices = model.get_sv_indices()
            
            Stage3 = (numofSV, svIndices)
            
            #################################################################################
            
            p_label, p_acc, p_val = svm_predict(list(yte),[list(row) for row in Test_K_mat], model, ('-b 1'))
            
            Perf_eva = EvaluateTest(yte, p_label)
            AucRoc   = computeRoc(p_label, p_val)
            
            Stage4 = (Perf_eva, AucRoc, p_label, p_val)
            #################################################################################
            
            Result[str(ker) +'_' + str(par)] = (Stage1, Stage2, Stage3, Stage4)
    
        Out[str(ker)] = Result
        
    return Out
    
    