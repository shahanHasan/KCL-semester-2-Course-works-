#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:47:53 2021

@author: Sangeeth
"""

import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics 
import seaborn as sns
import itertools



class K_Means_Clustering(object):
    
    def __init__(self):
        self.wholeSale = pd.read_csv("wholesale_customers.csv")
        #wholeSale.head(5)
        self.wholeSale_modified = self.wholeSale.drop(columns=['Channel','Region'])
        #wholeSale_modified.head(5)
        
    def question1(self,wholeSale_modified):
        desc = wholeSale_modified.describe()
        desc = desc.drop(["25%", "50%", "75%"])
        print(desc) 
        
    def question2_alternative(self,wholeSale_modified):
        n_clusters = 3
        kmeans,y_kmeans,centers,SSE,distSpace = self.KmeanClassifier(n_clusters,wholeSale_modified)
        Attributes_Dict,Combination = self.plot_data(wholeSale_modified)
        fig_num = 1
        
        for each in Combination:
            x = Attributes_Dict[each[0]]
            y = Attributes_Dict[each[1]]
            xlabel = each[0]
            ylabel = each[1]
            self.Scatter_Plot(fig_num,x,y,y_kmeans,xlabel,ylabel,centers)
            fig_num += 1
            
    def question2(self,wholeSale_modified):
        n_clusters = 3
        kmeans,y_kmeans,centers,SSE,distSpace = self.KmeanClassifier(n_clusters,wholeSale_modified)
        Attributes_Dict,Combination = self.plot_data(wholeSale_modified)
        #fig_num = 1
        
        fig, axs = plt.subplots(5,3, figsize=(50, 50), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 0.4, wspace=.001)
        
        
        for ax, each in zip(axs.ravel(), Combination):
            x = Attributes_Dict[each[0]]
            y = Attributes_Dict[each[1]]
            xlabel = each[0]
            ylabel = each[1]
            
            ax.scatter(x,y, c=y_kmeans, s=10, cmap='autumn')
            ax.scatter(centers[:, 0], centers[:, 1], c='black', s=10, alpha=0.5);
            #ax.grid()
            
            font = {'family':'serif','color':'black','size':5}
            ax.set_xlabel(xlabel,fontdict=font)
            ax.set_ylabel(ylabel,fontdict=font)
            #ax.set_title(f"{xlabel} vs {ylabel}",fontdict=font)
            ax.tick_params(axis='x', labelsize=5)
            ax.tick_params(axis='y', labelsize=5)

        plt.show()
            
        
    
    def KmeanClassifier(self,n_clusters,wholeSale_modified):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(wholeSale_modified)
        y_kmeans = kmeans.labels_
        centers = kmeans.cluster_centers_
        SSE = kmeans.inertia_
        distSpace = kmeans.transform(wholeSale_modified)
        
        return kmeans,y_kmeans,centers,SSE,distSpace
    
    def plot_data(self,wholeSale_modified):
        Fresh = wholeSale_modified.to_numpy()[:,0]
        Milk = wholeSale_modified.to_numpy()[:,1]
        Grocery = wholeSale_modified.to_numpy()[:,2]
        Frozen = wholeSale_modified.to_numpy()[:,3]
        Detergents_Paper = wholeSale_modified.to_numpy()[:,4]
        Delicassen = wholeSale_modified.to_numpy()[:,5]
        
        Attributes_Dict = {
            "Fresh" : Fresh,
            "Milk" : Milk , 
            "Grocery" : Grocery , 
            "Frozen" : Frozen ,
            "Detergents_Paper" : Detergents_Paper ,
            "Delicassen" : Delicassen 
         }
        ListofAttributes = ["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]
        Combination = list(itertools.combinations(ListofAttributes, 2))
        
        return Attributes_Dict,Combination
    
    def Scatter_Plot(self,fig_num,x,y,y_kmeans,xlabel,ylabel,centers):
        
        plt.figure(fig_num)
        #plt.subplot(16,1,fig_num)
        plt.gca()
        plt.scatter(x,y, c=y_kmeans, s=200, cmap='autumn')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
        plt.grid()
        font = {'family':'serif','color':'black','size':15}
        plt.title(f"{xlabel} vs {ylabel}",fontdict=font)
        plt.xlabel(xlabel,fontdict=font)
        plt.ylabel(ylabel,fontdict=font)
        #plt.legend()
        plt.show()
        
    def Question3(self,wholeSale_modified):
        """
        constants
        """
        n = len(wholeSale_modified)
        k = [3,5,10]
        K_set = [3,5,10]
        analytics = {}
        for i in K_set:
            self.k_means_algorithm_loop(i,wholeSale_modified,analytics)
        
        """
        k = 3
        """
        k_3_SSE = analytics[3]["SSE"]
        k_3_labels = analytics[3]["y_kmeans"]
        
        """
        k = 5
        """
        k_5_SSE = analytics[5]["SSE"]
        k_5_labels = analytics[5]["y_kmeans"]
        
        """
        k = 10
        """
        k_10_SSE = analytics[10]["SSE"]
        k_10_labels = analytics[10]["y_kmeans"]
        
        """
        CH
        """
        
        CH_3 = self.get_CH(wholeSale_modified,k_3_labels)
        CH_5 = self.get_CH(wholeSale_modified,k_5_labels)
        CH_10 = self.get_CH(wholeSale_modified,k_10_labels)
        
        """
        BC
        """
        
        K_3_BC = self.get_BC(CH_3,k_3_SSE,440,3)
        
        K_5_BC = self.get_BC(CH_5,k_5_SSE,440,5)
        K_10_BC = self.get_BC(CH_10,k_10_SSE,440,10)
        # print(K_3_BC)
        # print(K_5_BC)
        # print(K_10_BC)
        
        k_3_ratio = K_3_BC/k_3_SSE
        k_5_ratio = K_5_BC/k_5_SSE
        k_10_ratio = K_10_BC/k_10_SSE
        
        
        data = np.array([
            [K_3_BC,K_5_BC,K_10_BC],
            [k_3_SSE,k_5_SSE,k_10_SSE],
            [k_3_ratio,k_5_ratio,k_10_ratio],
            [CH_3,CH_5,CH_10]
        ])
        
       
        plt.figure(figsize=(5,5))
        x_axis_labels = ["k = 3","k = 5", "k = 10"]
        y_axis_labels = ["BC","WC","BC / WC", "CH index"]
        sns.heatmap(data=data,linewidths=.5, annot=True,square = True,  cmap = 'Purples',xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        plt.title("Table of analytics :")
        plt.show()

        
        
    def k_means_algorithm_loop(self,num,data,analytics):
        kmeans = KMeans(n_clusters=num, random_state=0).fit(data)
        y_kmeans = kmeans.labels_
        centers = kmeans.cluster_centers_
        SSE = kmeans.inertia_
        analytics[num] = {
            "y_kmeans" : y_kmeans,
            "centers" : centers,     
            "SSE" : SSE        
        }
        
    
    def get_CH(self,data,labels):
        CH = metrics.calinski_harabasz_score(data, labels) 
        return CH
    
    def get_BC(self,ch,sse,n,k):
        BC = (ch*sse*(k-1))/(n-k)
        return BC
        
    def main(self):
        print()
        print("------------------- Question 1 -------------------------------")
        print()
        
        self.question1(self.wholeSale_modified)
        
        print()
        print("------------------- Question 2 -------------------------------")
        print()
        
        self.question2(self.wholeSale_modified)
        
        print()
        print("------------------- Question 2 alternative -------------------------------")
        print()
        
        self.question2_alternative(self.wholeSale_modified)
        
        print()
        print("------------------- Question 3 -------------------------------")
        print()
        
        self.Question3(self.wholeSale_modified)
        

if __name__ == "__main__":
    
    k = K_Means_Clustering()
    k.main()
    
    
