#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 01:55:38 2021

@author: shahan
"""
from sklearn import datasets
import numpy as np
from sklearn.decomposition import sparse_encode

class Sparse_Encoding(object):
    
    """Global Variables"""
    iris = datasets.load_iris()
    iris_Feature_vectors = iris.data
    iris_Class_labels = iris.target
    
    def __init__(self):
        self.Dict = {}
        self.empty1 = np.empty((0,4))
        self.empty2 = np.empty((0,4))
        self.empty3 = np.empty((0,4))
        self.Sample_Features = np.array([
            [7.0,2.2,2.4,2.1],
            [6.5,2.1,2.1,1.2],
            [7.1,2.7,1.8,0.8],
            [6.8,2.4,5.0,0.6],
            [5.2,3.8,3.8,1.3],
        ])

    def Dictionery(self,iris_Feature_vectors,iris_Class_labels,empty1,empty2,empty3,Dict):
        
        for (iris_Feature_vector, iris_Class_label) in zip(iris_Feature_vectors, iris_Class_labels):
            #print(f"{iris_Feature_vector} : {iris_Class_label}")
            if(iris_Class_label == 0):
                #print(iris_Feature_vector)
                empty1 = np.vstack((empty1,iris_Feature_vector))
                Dict[iris_Class_label] = empty1
        
            if(iris_Class_label == 1):
                #print(iris_Feature_vector)
                empty2 = np.vstack((empty2,iris_Feature_vector))
                Dict[iris_Class_label] = empty2
            
            if(iris_Class_label == 2):
                #print(iris_Feature_vector)
                empty3 = np.vstack((empty3,iris_Feature_vector))
                Dict[iris_Class_label] = empty3
                
        return Dict
    
    def Cost(self,Sample_Features, Dict, algorithm='omp',numNonZero = 2, alpha=1.000000e-05, lambdaVal = 0.1):
        y=sparse_encode(Sample_Features.reshape(1,-1),Dict,algorithm=algorithm,n_nonzero_coefs=numNonZero,alpha=alpha)
        Dict_transpose = np.transpose(Dict)
        y_transpose = np.transpose(y)
        product = Dict_transpose @ y_transpose
        #print(product)
        y_l0norm = len(np.nonzero(y_transpose))
        Distribution = Sample_Features.reshape(-1,1) - product
        #print(Distribution)
        cost = np.linalg.norm(Distribution) + (y_l0norm * lambdaVal )
        return cost
                
    
    def main(self):
        
        print("Make Dictionery")
        Dict = self.Dictionery(self.iris_Feature_vectors,self.iris_Class_labels,self.empty1,self.empty2,self.empty3, self.Dict)
        print("Calculate Cost : ")
        
        for i in Dict:
            if(i == 0):
                print()
                print("Class 0 costs")
                print()
            if(i == 1):
                print()
                print("Class 1 costs")
                print()
            if(i == 2):
                print()
                print("Class 2 costs")
                print()
            for j in self.Sample_Features:
                #print(Dict[i])
                cost = self.Cost(j, Dict[i])
                print(np.round(cost,4))
        
if __name__ == '__main__':
    
    Sparse_encode =  Sparse_Encoding()
    Sparse_encode.main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        