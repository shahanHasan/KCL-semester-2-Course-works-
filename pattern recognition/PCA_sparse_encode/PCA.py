#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 01:26:47 2021

@author: shahan
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

class PCA_ML(object):
    """Global Vars"""
    iris = datasets.load_iris()
    iris_Feature_vectors = iris.data
    iris_Class_labels = iris.target
    Class_Label_names = iris.target_names
    Description = iris.DESCR
    Feature_Names = iris.feature_names
    
    def __init__(self):
        """Object Specific Macros setup """
        self.Sample_dict = np.array([
    [5.2,2.7,3.3,0.6],
    [5.1,2.3,6.3,1.6],
    [7.9,2.2,3.4,1.2],
    [6.0,3.8,3.3,0.3],
    [5.2,3.5,4.9,1.0],
])
        
    def PCA_training(self, iris_Feature_vectors, n_components=2 ):
        pca = PCA(n_components=2)
        pca.fit(iris_Feature_vectors)
        X_new = pca.transform(iris_Feature_vectors)
        return pca , X_new

    def plot(self,Class_Label_names,X_new,iris_Class_labels):
        plt.figure()
        colors = ['navy', 'turquoise', 'darkorange']
        lw = 2
        
        for color, i, Class_Label_name in zip(colors, [0, 1, 2], Class_Label_names):
            plt.scatter(X_new[iris_Class_labels == i, 0], X_new[iris_Class_labels == i, 1], color=color, alpha=.8, lw=lw,
                        label=Class_Label_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of IRIS dataset')
        
    def PCA_Test(self, pca):
        NewPCASamples = pca.transform(self.Sample_dict)
        return np.round(NewPCASamples,4)
    
    def main(self):
        print("PCA Train ")
        print()
        pca , X_new = self.PCA_training(self.iris_Feature_vectors)
        print()
        print("plot")
        print()
        self.plot(self.Class_Label_names, X_new, self.iris_Class_labels)
        print()
        print("PCA Test")
        print()
        NewPCASamples = self.PCA_Test(pca)
        print(NewPCASamples)
        
if __name__ == '__main__':
    
    PCA_new =  PCA_ML()
    PCA_new.main()
    
