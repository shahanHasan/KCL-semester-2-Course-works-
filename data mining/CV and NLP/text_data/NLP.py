#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:47:45 2021

@author: Sai
"""
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import flatten
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import time

import pickle
import os
from os import path

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class NLP(object):
    
    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.df = pd.read_csv("Corona_NLP_train.csv")
        self.AugmentedData = self.df.copy()
        self.Task4DataSet = self.df.copy()
        self.WNL = WordNetLemmatizer()
        
        self.filename = "Data_Saved"
        
        # Check if file exits 
        if(not path.isfile(self.filename)):
            # IF does not Exist, Create
            self.f = open(self.filename, "x")
        
        # Create a new file if does not exist
        
        # self.f = open(self.filename, "w")
        
        if (not os.stat(self.filename).st_size == 0):
            infile = open(self.filename,'rb')
            self.SavedData = pickle.load(infile)
            infile.close()
        
        self.data = {}
        
        if not os.path.exists('NLP_Graphs'):
            os.makedirs('NLP_Graphs')
        
        
    def Task_1_Question_1(self):
        
        if (os.stat(self.filename).st_size == 0):
            
            
            # Part 1 
            SentimentTypes = set(self.df["Sentiment"])
            print(f"Types of sentiments : {SentimentTypes}")
            countNum = 0
            sentimentDict = {}
            
            for sentiment in SentimentTypes:
                num = self.df['Sentiment'].loc[lambda sentimentInner: sentimentInner == sentiment].count()
                countNum += num
                sentimentDict[sentiment] = num
                
            SecondLastValue = list(sorted(sentimentDict.items(), key = lambda kv:(kv[1], kv[0])))[-2]
           
            
            # Part 2
            
            Sentiment_TweetAT_df = self.df[["Sentiment","TweetAt"]].loc[self.df["Sentiment"] == "Extremely Positive"]
            dates = set(self.df[["Sentiment","TweetAt"]].loc[self.df["Sentiment"] == "Extremely Positive"]["TweetAt"])
            date_Dict = {}
            
            for date in dates:
                dateNum = Sentiment_TweetAT_df['TweetAt'].loc[lambda dateTweeted : dateTweeted == date].count()
                date_Dict[date] = dateNum
            
            dateMaxPositiveTweets = sorted(date_Dict.items(), key = lambda kv:(kv[1], kv[0]))[-1]
           
            
            
            # Part 3
            
            self.AugmentedData["OriginalTweet"] = self.AugmentedData["OriginalTweet"].apply(self.__text_process)
            
            self.data["SecondLastValue"] = SecondLastValue
            self.data["dateMaxPositiveTweets"] = dateMaxPositiveTweets
            
        else:
                 
            SecondLastValue = self.SavedData["SecondLastValue"]
            dateMaxPositiveTweets = self.SavedData["dateMaxPositiveTweets"]
    

        
        return SecondLastValue , dateMaxPositiveTweets 
    
    def __text_process(self,data):
        msg=[w.lower() for w in data if w.isalpha() or w.isspace()]
        msg=''.join(msg)
        return msg
    
    
    def Task_1_Question_2(self):
        
        
        
        if (os.stat(self.filename).st_size == 0):
            
            self.AugmentedData["OriginalTweet"] = self.AugmentedData["OriginalTweet"].apply(self.__tokenize)
            
            # With Stop Words
            
            # numPerRow = self.AugmentedData["OriginalTweet"].apply(self.__numberOfWords)
            # num = sum(numPerRow)
            
            TweetsWithAllWords = self.AugmentedData["OriginalTweet"].to_list()
            newFlattenedList = list(flatten(TweetsWithAllWords))
            
            totalNumberofallWords = len(newFlattenedList)
            totalNumberofallDistinctWords = len(set(newFlattenedList))
            
            
            c = Counter(newFlattenedList)
            MostCommonWords_10 = c.most_common(10)
            
            
            
            # Removing Stop Words and letter less than or 2 letters
            
            newListExcludingStopWords = [w for w in newFlattenedList if w not in self.stopwords]
            newListExcludingStopWordsAndLessThan2Letter = [w for w in newListExcludingStopWords if len(w) > 2]
    
    
            # Without Stop words and words <= 2 letters
            
            totalNumberofallNewWords = len(list(flatten(newListExcludingStopWordsAndLessThan2Letter)))
            totalNumberofallNewDistinctWords = len(set(list(flatten(newListExcludingStopWordsAndLessThan2Letter))))
            
            c2 = Counter(newListExcludingStopWordsAndLessThan2Letter)
            MostCommonWordsNewCorpus_10 = c2.most_common(10)
            
            # Freq Dist Plots
            
            fd = nltk.FreqDist(newFlattenedList)
            
            
            fd_No_Stopwords = nltk.FreqDist(newListExcludingStopWordsAndLessThan2Letter)
            
            Head = self.AugmentedData["OriginalTweet"].head()
            
            self.data["totalNumberofallWords"] = totalNumberofallWords
            self.data["totalNumberofallDistinctWords"] = totalNumberofallDistinctWords
            self.data["MostCommonWords_10"] = MostCommonWords_10
            self.data["totalNumberofallNewWords"] = totalNumberofallNewWords
            self.data["totalNumberofallNewDistinctWords"] = totalNumberofallNewDistinctWords
            self.data["MostCommonWordsNewCorpus_10"] = MostCommonWordsNewCorpus_10
            self.data["c2"] = c2
            self.data["fd"] = fd
            self.data["fd_No_Stopwords"] = fd_No_Stopwords
            self.data["Head"] =  Head
            
        else:
            
            totalNumberofallWords = self.SavedData["totalNumberofallWords"]
            totalNumberofallDistinctWords = self.SavedData["totalNumberofallDistinctWords"]
            MostCommonWords_10 = self.SavedData["MostCommonWords_10"]
            totalNumberofallNewWords = self.SavedData["totalNumberofallNewWords"]
            totalNumberofallNewDistinctWords = self.SavedData["totalNumberofallNewDistinctWords"]
            MostCommonWordsNewCorpus_10 = self.SavedData["MostCommonWordsNewCorpus_10"]
            c2 = self.SavedData["c2"]
            fd = self.SavedData["fd"]
            fd_No_Stopwords = self.SavedData["fd_No_Stopwords"]
            Head = self.SavedData["Head"]
            
         
        plt.ion()
        # fig = plt.figure(num = "With Stopwords", figsize = (10,4))
        # plt.gcf().subplots_adjust(bottom=0.15) # to avoid x-ticks cut-off
        plt.title("With StopWords")
        fd.plot(10,cumulative=False)
        plt.savefig("NLP_Graphs/FreqDistWithStopWords.jpeg")
        plt.ioff()
      
        
        plt.ion()
        # fig = plt.figure(num = "Without Stopwords" , figsize = (10,4))
        # plt.gcf().subplots_adjust(bottom=0.15) # to avoid x-ticks cut-off
        plt.title("Without Stop Words and words <= 2 letters")
        fd_No_Stopwords.plot(10,cumulative=False)
        plt.savefig("NLP_Graphs/FreqDistWithoutStopWords.jpeg")
        plt.ioff()
    
        
            
        
        return Head , c2 , totalNumberofallWords , totalNumberofallDistinctWords , MostCommonWords_10 , totalNumberofallNewWords , totalNumberofallNewDistinctWords , MostCommonWordsNewCorpus_10
        
        
        
        
    def __numberOfWords(self,data):
        return len(data)
    
    def __tokenize(self, data):
        msg = nltk.word_tokenize(data)
        return msg
    
    
    def Task_1_Question_3(self,counter):
        """
        Observation :

            Words of the new corpus are semantically more relevant to the dataset and the problem.
            Noise reduction, irrelevance reduction, Increased Efficiency of the dataset
            
        """
        
        if (os.stat(self.filename).st_size == 0):
        
            MostCommonWords = counter.most_common()
    
            x =0
            y =0
            x_axis = {}
            y_axis = {}
            
            # Data Preparation 
            
            for i in range(10):
                y += 100
                #print(x,y)
                x_axis_l = []
                y_axis_l = []
                for word in MostCommonWords[x:y]:
                    #print(word)
                    x_axis_l.append(word[0])
                    count_documents = self.AugmentedData["OriginalTweet"].apply(self.__counter,word=word[0]).count()
                    y_axis_l.append(count_documents)
               
                x_axis[i] = x_axis_l
                y_axis[i] = y_axis_l
                if(i == 0):
                    x += 101
                else: 
                    x += 100 
                    
            self.data["x_axis"] = x_axis
            self.data["y_axis"] = y_axis
        
        else:
            
            x_axis = self.SavedData["x_axis"]
            y_axis = self.SavedData["y_axis"]
                
        # Graphs
        
        for i,j in zip(x_axis,y_axis):
                

            fig = plt.figure(figsize=(18,7))
            line = sns.lineplot(x=x_axis[i], y=y_axis[j])
            plt.xticks(rotation=90)
            plt.xlabel("Words")
            plt.ylabel("Count of Word Occurences")
            plt.legend(["Number of Documents Containing the Word"],framealpha=1, frameon=True)
            plt.grid(True)
            plt.savefig(f"NLP_Graphs/Document_Word_Counts_{i}.jpeg")
            plt.close(fig)
            
        
        
    def __counter(self,data, word):
        if word in data:
            return True

    
    def Task_1_Question_4(self):
        
        if (os.stat(self.filename).st_size == 0):
            
            
            sample = self.Task4DataSet["OriginalTweet"]
            sample_target = self.Task4DataSet["Sentiment"]
            
            # Splitting the data - 70:30 ratio
            X_train, X_test, y_train, y_test = train_test_split(sample , sample_target, test_size = 0.3, random_state = 99)
            
            
            # Count Vector Train
            CV=CountVectorizer(analyzer=self.__text_process_Task_4)
            CV.fit(X_train)
            
            # Transform Train 
            X_train=CV.transform(X_train)
            # Transform Test
            X_test = CV.transform(X_test)
            
            # TFIDF
            Tfidf=TfidfTransformer()
            Tfidf.fit(X_train)
            # Train and Test transform
            Tfidf_val=Tfidf.transform(X_train)
            X_test=Tfidf.transform(X_test)
            
            
            # Classifier
            sentiment=MultinomialNB()
            sentiment.fit(Tfidf_val,y_train)
            result1=sentiment.predict(X_test)
            
            
            
            self.data["y_test"] = y_test
            self.data["result1"] = result1
            
            
        else:
            
            y_test = self.SavedData["y_test"]
            result1 = self.SavedData["result1"]
        
        print(classification_report(y_test,result1))
            
        cf = confusion_matrix(y_test, result1 , labels= y_test.unique())
        
        Error_Rate , Accuracy_Score = self.__ErrorRate_Accuracy(cf)
        print(f"Error Rate : {Error_Rate}")
        print(f"Accuracy Score : {Accuracy_Score}")
        
        
        # Heat Map Plot
        fig = plt.figure("Heat Map Confusion Matrix" , figsize=(5,5))
        HeatMap = sns.heatmap(data=cf,linewidths=.5, annot=True,square = True,  cmap = "twilight_shifted")
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        Error_Rate = round((Error_Rate*100), 2)
        all_sample_title = f'Error rate: {Error_Rate} %'
        plt.title(all_sample_title, size = 15)
        plt.savefig('NLP_Graphs/ConfusionMatrix.jpeg')
        plt.close(fig)
        
        # Comparison
        df1=pd.DataFrame()
        df1['actual']=y_test
        df1['predicted']=result1
        print("Comparison between Actual and Predicted Sentiments")
        print()
        print(df1.head())
        
        
        
        
    def __ErrorRate_Accuracy(self, cf):
        # Class 0 
        
        TP_0 = cf[0,0]
        FP_0 = cf[0,1] + cf[0,2] + cf[0,3] + cf[0,4]
        cf_0 = np.delete(cf, 0 , 1)
        cf_0 = np.delete(cf_0, 0 , 0)
        TN_0 = np.sum(cf_0)
        FN_0 = cf[:,0].sum() - TP_0
        
        # Class 1 
        
        TP_1 = cf[1,1]
        FP_1 = cf[1,0] + cf[1,2] + cf[1,3] + cf[1,4]
        cf_1 = np.delete(cf, 1 , 1)
        cf_1 = np.delete(cf_1, 1 , 0)
        TN_1 = np.sum(cf_1)
        FN_1 = cf[:,1].sum() - TP_1
        
        # Class 2
        
        TP_2 = cf[2,2]
        FP_2 = cf[2,0] + cf[2,1] + cf[2,3] + cf[2,4]
        cf_2 = np.delete(cf, 2 , 1)
        cf_2 = np.delete(cf_2, 2 , 0)
        TN_2 = np.sum(cf_2)
        FN_2 = cf[:,2].sum() - TP_2
        
        # Class 3
        
        TP_3 = cf[3,3]
        FP_3 = cf[3,0] + cf[3,1] + cf[3,2] + cf[3,4]
        cf_3 = np.delete(cf, 3 , 1)
        cf_3 = np.delete(cf_3, 3 , 0)
        TN_3 = np.sum(cf_3)
        FN_3 = cf[:,3].sum() - TP_2
        
        # Class 4
        
        TP_4 = cf[4,4]
        FP_4 = cf[4,0] + cf[4,1] + cf[4,2] + cf[4,3]
        cf_4 = np.delete(cf, 4 , 1)
        cf_4 = np.delete(cf_4, 4 , 0)
        TN_4 = np.sum(cf_4)
        FN_4 = cf[:,4].sum() - TP_2
        
        TP = TP_0 + TP_1 + TP_2 + TP_3 + TP_4
        FP = FP_0 + FP_1 + FP_2 + FP_3 + FP_4
        TN = TN_0 + TN_1 + TN_2 + TN_3 + TN_4
        FN = FN_0 + FN_1 + FN_2 + FN_3 + FN_4
        
        
        Error_Rate = (FP+FN) / (TN+FP+FN+TP)
        Accuracy_Score = 1 - Error_Rate
        
        
        return Error_Rate , Accuracy_Score
    
    def __text_process_Task_4(self,data):
        msg = [w.lower() for w in data if w.isalpha() or w.isspace()]
        msg = ''.join(msg)
        msg = [w for w in msg.split() if w not in self.stopwords]
        msg = [w for w in msg if len(w) > 2]
        msg = [self.WNL.lemmatize(word) for word in msg ]
        #msg = nltk.word_tokenize(data)
        return msg

    
    def main(self):
        print()
        print("------------------- Question 1.1 -------------------------------")
        print()
        
        SecondLastValue , dateMaxPositiveTweets  = self.Task_1_Question_1()
        
        print(f"Second most popular sentiment : {SecondLastValue}")
        print()
        print(f"Date with the greatest number of extremely positive tweets : {dateMaxPositiveTweets}")
        print()
        print("Transformed Data : ")
        print(self.AugmentedData["OriginalTweet"].head())
        
        print()
        print("------------------- Question 1.2 -------------------------------")
        print()
        
        Head , c2 , totalNumberofallWords , totalNumberofallDistinctWords , MostCommonWords_10 , totalNumberofallNewWords , totalNumberofallNewDistinctWords , MostCommonWordsNewCorpus_10 = self.Task_1_Question_2()
        print("Tokenized Data : ")
        print(Head)
        print()
        print(f"Total number of all words : {totalNumberofallWords}")
        print(f"Total number of all Distinct words : {totalNumberofallDistinctWords}")
        print(f"10 Most frequent words in the corpus : {MostCommonWords_10}")
        
        print()
        print("Modifying Corpus")
        print()
        
        print(f"Total number of all words in the modified Corpus :  {totalNumberofallNewWords}")
        print(f"Total number of all Distinct words in the modified Corpus : {totalNumberofallNewDistinctWords}")
        print(f"10 Most frequent words in the modified corpus : {MostCommonWordsNewCorpus_10}")
        
        print()
        print("------------------- Question 1.3 -------------------------------")
        print()
        
        print("Graphs : ")
        
        self.Task_1_Question_3(c2)
        
        print()
        print("------------------- Question 1.4 -------------------------------")
        print()
        
        self.Task_1_Question_4()
        
        if (os.stat(self.filename).st_size == 0):    
            outfile = open(self.filename,'wb')
            pickle.dump(self.data,outfile)
            outfile.close()
            
    
    
 
if __name__ == "__main__":

    
    
    start_time = time.time()

    NLP = NLP()
    NLP.main()
    print()
    print(f"--- {time.time() - start_time} seconds ---")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
