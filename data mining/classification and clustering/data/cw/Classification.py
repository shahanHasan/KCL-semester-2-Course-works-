import pandas as pd
import numpy as np
from sklearn import preprocessing
import pprint

import matplotlib.pyplot as plt
# %matplotlib inline #for encoding
from sklearn.model_selection import train_test_split 
#for decision tree object
from sklearn.tree import DecisionTreeClassifier 
#for checking testing results
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#for visualizing tree 
from sklearn.tree import plot_tree

class Classification_descisionTree(object):
    
    
    def __init__(self):
        """
        Global Variables
        """
        self.df = pd.read_csv("adult.csv")
        self.df_New = self.df.drop(columns=['fnlwgt'])
        self.ClassLabels = self.df_New["class"]
        # Drop class from dataset
        self.df_New_noClass = self.df_New.drop(columns=['class'])
        # df_New_noClass.head()
        self.row , self.col = self.df_New_noClass.shape
        self.total_number_of_attributes = self.row * self.col
        # Easier way :
        self.nullValuesEasy = self.df_New.isnull().sum().sum()

    def Question1_i(self,data):
        
        Number_of_instances = len(data.index)
        return Number_of_instances
        
    def Question1_ii(self,data):
        
        # Hard Way :
        nullValues = 0
        null_value_object = data.isnull().sum()
        # print(type(null_value_object))
        for i in null_value_object:
            if (i != 0 ) :
                nullValues += i
        
        
        return nullValues
        
    
    def Question1_iii(self,nullValuesEasy,total_number_of_attributes):
    
        Fraction_Of_missing_Values = nullValuesEasy / total_number_of_attributes  
        return Fraction_Of_missing_Values
        
        
    def Question1_iv(self,data):
        #Here data is datset with no class column
        count = 0
        for row in range(len(data)):
            for key in data.loc[row]:
                if (pd.isna(key)):
                    count+=1
                    break
    
        
        return count

        
    def Question1_v(self, count, Number_of_instances):
        #Pass in count , question 4.iv
        Fraction_of_instances_with_missingValues_OverAll = count / Number_of_instances
        
        return Fraction_of_instances_with_missingValues_OverAll
    
    def descritization(self,data):
        Label_list = {}
        LabelEncoder = preprocessing.LabelEncoder()
        for label in data.columns.values:
            if(data[label].dtypes == int or data[label].dtypes == str):
                LabelEncoder.fit(data[label])
                Label_list[label] = list(LabelEncoder.classes_)
            else :
                LabelEncoder.fit(data[label].astype(str))
                Label_list[label] = list(LabelEncoder.classes_)
        
        return Label_list
    
    def Question2(self,data):
        
        labels = self.descritization(data)
        #pprint.pprint(labels)
        return labels
        
        
    def main(self):
        
        print()
        print("------------------- Question 1 -------------------------------")
        print()
        
        print(f"Total number of attributes : {self.total_number_of_attributes}")
        
        #question 1.i
        Number_of_instances = self.Question1_i(self.df_New)
        print(f"Number of instances : {Number_of_instances}")
        
        #question 1.ii
        nullValues = self.Question1_ii(self.df_New)
        print(f"Number of null values : {nullValues}")
        
        #question 1.iii
        Fraction_Of_missing_Values = self.Question1_iii(self.nullValuesEasy, self.total_number_of_attributes)
        print(f"Fraction of missing values over all attributes : {Fraction_Of_missing_Values}")
        
        #question 1.iv
        count = self.Question1_iv(self.df_New_noClass)
        print(f"Number of instances with missing values : {count}")
        
        #question 1.v
        Fraction_of_instances_with_missingValues_OverAll = self.Question1_v(count,Number_of_instances)
        print(f"Fraction of instances with missing values over all instances : {Fraction_of_instances_with_missingValues_OverAll}")
        
        print()
        print("------------------- Question 2 -------------------------------")
        print()
        
        labels = self.Question2(self.df_New_noClass)
        pprint.pprint(labels)
        
        
        print()
        print("------------------- Question 3 -------------------------------")
        print()
        
        #Data Processing 
        X,y = self.dataPreProcessing_Q_3(self.ClassLabels,self.df)
        print(self.ClassLabels.unique())
        print("-------------------")
        print(f"Encoded Labels : {y}")
        
        #Data Splitting
        X_train, X_test, y_train, y_test = self.dataSplitting(X,y)
        print(f"Training split input: {X_train.shape}")
        print(f"Testing split input : {X_test.shape}")
        
        #Descision Tree Model Building
        print("Creating a Descision tree Classifier!")
        dtree = self.DtreeClassifier(X_train,y_train)
        print("--------------------------------")
        print('Decision Tree Classifier Created')
        
        #Predictions
        y_pred_default = self.predict(dtree,X_test)
        
        
        #Model Evaluation
        report = self.ClassificationReport(y_pred_default,y_test)
        print(report)
        
        #Error Rate And Accuracy 
        tn, fp, fn, tp, Error_Rate = self.Confusion_Matrix_Error_Rate(y_pred_default,y_test)
        print(f"tn : {tn}")
        print(f"fp : {fp}")
        print(f"fn : {fn}")
        print(f"tp : {tp}")
        
        print(f"Error Rate : {Error_Rate}")
        print(f"Accuracy Score : {1-Error_Rate}")
        
        
        print()
        print("------------------- Question 4 -------------------------------")
        print()
        
        #Data Processing and DataSplitting
        D_prime = self.D_prime(self.df_New)
        
        D_one_prime,D_one_prime_class = self.D_one_prime(D_prime)
        D_two_prime,D_two_prime_class = self.D_two_prime(D_prime,self.df_New_noClass)
        
        X_one_prime,y_one_prime,X_two_prime,y_two_prime,X_one_prime_test,y_one_prime_test,X_two_prime_test,y_two_prime_test = self.D_prime_data_splitting(
                                                 D_one_prime,D_one_prime_class,
                                                 D_two_prime,D_two_prime_class,
                                                 self.df_New,self.df_New_noClass)
        
        X_one_prime,y_one_prime_encode,X_two_prime,y_two_prime_encode,X_one_prime_test,y_one_prime_encode_test,X_two_prime_test,y_two_prime_encode_test = self.D_prime_data_preprocessing_splitting(
            X_one_prime,y_one_prime,
            X_two_prime,y_two_prime,
            X_one_prime_test,y_one_prime_test,
            X_two_prime_test,y_two_prime_test,
            self.ClassLabels)
        
        print(f"Training split 1 input: {X_one_prime.shape}")
        print(f"Training split 2 input: {X_two_prime.shape}")
        print(f"Testing split 1 input : {X_one_prime_test.shape}")
        print(f"Testing split 2 input : {X_two_prime_test.shape}")
        
        #Descision Tree Model Building
        d_one_prime_tree = self.DtreeClassifier(X_one_prime,y_one_prime_encode)
        
        print("--------------------------------")
        print('Decision Tree Classifier 1 Created')
        print("--------------------------------")
        print()

        d_two_prime_tree = self.DtreeClassifier(X_two_prime,y_two_prime_encode)
        
        print("--------------------------------")
        print('Decision Tree Classifier 2 Created')
        print("--------------------------------")
        print()
        
        #Predictions
        # making predictions
        y_one_prime_predict = self.predict(d_one_prime_tree,X_one_prime_test)
        y_two_prime_predict = self.predict(d_two_prime_tree,X_two_prime_test)
        
        #Model Evaluation
        print()
        report_D_one_prime = self.ClassificationReport(y_one_prime_predict,y_one_prime_encode_test)
        print("Printing Classification Report : D one prime : ")
        print(report_D_one_prime)
        
        print()
        report_D_two_prime = self.ClassificationReport(y_two_prime_predict,y_two_prime_encode_test)
        print("Printing Classification Report : D two prime : ")
        print(report_D_two_prime)
        
        
        #Error Rate And Accuracy 
        print("Error Rate and accuracy of D one Prime")
        tn, fp, fn, tp, Error_Rate = self.Confusion_Matrix_Error_Rate(y_one_prime_encode_test,y_one_prime_predict)
        print(f"tn : {tn}")
        print(f"fp : {fp}")
        print(f"fn : {fn}")
        print(f"tp : {tp}")
        Accuracy_D_one_prime = 1 - Error_Rate
        print(f"Error Rate : {Error_Rate}")
        print(f"Accuracy Score : {1-Error_Rate}")
        
        #Error Rate And Accuracy 
        print("Error Rate and accuracy of D two Prime")
        tn, fp, fn, tp, Error_Rate = self.Confusion_Matrix_Error_Rate(y_two_prime_encode_test,y_two_prime_predict)
        print(f"tn : {tn}")
        print(f"fp : {fp}")
        print(f"fn : {fn}")
        print(f"tp : {tp}")
        Accuracy_D_two_prime = 1 - Error_Rate
        print(f"Error Rate : {Error_Rate}")
        print(f"Accuracy Score : {1-Error_Rate}")
        
        print()
        print(f"Difference in accuracy of D one prime classifier and D two prime classifier : {abs(Accuracy_D_one_prime-Accuracy_D_two_prime)}")
        
        
        
        
    def __descritization(self,data):
        # encode categorical variables using label Encoder
        # select all categorical variables
    
        df_Encoded = data.drop(columns=['fnlwgt','class'])
        df_categorical = df_Encoded.select_dtypes(include=['object'])
        le = preprocessing.LabelEncoder()
        df_categorical = df_categorical.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
        # Next, Concatenate df_categorical dataframe with original df (dataframe)
        # first, Drop earlier duplicate columns which had categorical values
    
        df_Encoded = df_Encoded.drop(df_categorical.columns,axis=1)
        df_Encoded = pd.concat([df_Encoded,df_categorical],axis=1)
        #df_Encoded.head()
        return df_Encoded
        
    def dataPreProcessing_Q_3(self,ClassLabels,data):
        # Target Values, Class Labels , Y 
        LabelEncoder = preprocessing.LabelEncoder()
        y = LabelEncoder.fit_transform(ClassLabels)
        
        # Feature vector / Attributes / X
        df_Encoded = self.__descritization(data)
        X = df_Encoded
        return X,y
    
    def dataSplitting(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.3, random_state = 99)
        return X_train, X_test, y_train, y_test
    
    def DtreeClassifier(self,X_train,y_train):
        # Modeling Tree and testing it
        # Defining the decision tree algorithmdtree=DecisionTreeClassifier()

        dtree=DecisionTreeClassifier(max_depth=5)
        dtree.fit(X_train,y_train)
        return dtree
    
    def predict(self,dtree,X_test):
        # making predictions
        y_pred_default = dtree.predict(X_test)
        return y_pred_default
    
    def ClassificationReport(self,y_pred_default,y_test):
        report = classification_report(y_test,y_pred_default)
        return report
    
    def Confusion_Matrix_Error_Rate(self,y_pred_default,y_test ):
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()
        Error_Rate = (fp+fn) / (tn+fp+fn+tp)
        return tn , fp, fn, tp , Error_Rate
    
    def D_prime(self,data):
        # question 4 i
        # New dataset with only missing values :
            
        D_prime_null = data[data.isna().any(axis=1)]
        D_prime_null.head()
        
        #print(len(D_prime_null))
        # print(D_prime_null.info)
        
        # question 4 ii
        # filter out all rows containing one or more missing values 
        D_prime_notnull = data.dropna()
        D_prime_notnull.head()
        
        # Select x number of rows randomly
        D_prime_notnull_eq = D_prime_notnull.sample(n=3620, random_state=1)
        D_prime_notnull_eq.head()
        
        #print(len(D_prime_notnull_eq))
        # print(D_prime_notnull_eq.info)
        
        # Merge the 2 data sets
        # D_prime = pd.merge(D_prime_null, D_prime_notnull_eq, how='outer')
        D_prime = pd.concat([D_prime_null,D_prime_notnull_eq], axis=0)
        
        #print(len(D_prime))
        # D_prime.head()
        return D_prime
    
    def D_one_prime(self,D_prime):
        """
        Fill missing with "Missing " String
        """
        # D one prime and D 2 prime initialisation

        D_one_prime = D_prime.fillna("missing")
        # D_one_prime.head()
        # print(D_prime.isna().any())
        
        # Get Class Labels 
        D_one_prime_class = D_one_prime["class"]
        # D_one_prime_class.head()
        
        #Drop class column
        D_one_prime = D_one_prime.drop(columns=["class"])
        # D_one_prime.head()
        return D_one_prime,D_one_prime_class
    
    def D_two_prime(self,D_prime,data):
        """
        Fill missing with most popular attribute of the column
        """
        D_two_prime = D_prime.fillna(data.mode().iloc[0])
        # D_two_prime.head()

        # D prime class labels :
        # Get Class Labels 
        D_two_prime_class = D_two_prime["class"]
        # D_two_prime_class.head()
        
        #Drop class column
        D_two_prime = D_two_prime.drop(columns=["class"])
        # D_two_prime.head()
        return D_two_prime,D_two_prime_class
    
    def D_prime_data_splitting(self,D_one_prime,D_one_prime_class,
                               D_two_prime,D_two_prime_class,
                               df_New,df_New_noClass
                               ):
        # Feature Vector X and class label y init for d prime datasets
        # D-1-prime
        
        X_one_prime = D_one_prime
        y_one_prime = D_one_prime_class
        
        # D-2-prime
        
        X_two_prime = D_two_prime
        y_two_prime = D_two_prime_class
        
        # Test data set for d one prime
        
        # 30 % of original data set for testing
        X_one_prime_test = df_New.sample(frac=0.3, random_state=1)
        X_one_prime_test = X_one_prime_test.fillna("missing")
        
        y_one_prime_test = X_one_prime_test["class"]
        # print(y_one_prime_test)
        
        #X_one_prime_test.head()
        #print(len(X_one_prime_test))
        
        # Test data set for d two prime
        
        # 30 % of original data set for testing
        X_two_prime_test = df_New.sample(frac=0.3, random_state=1)
        X_two_prime_test = X_two_prime_test.fillna(df_New_noClass.mode().iloc[0])
        
        y_two_prime_test = X_two_prime_test["class"]
        # print(y_two_prime_test)
        
        #print(len(X_two_prime_test))
        #X_one_prime_test.head()
        
        # Drop and recreate x prime test instances
        X_one_prime_test = X_one_prime_test.drop(columns=["class"])
        X_two_prime_test = X_two_prime_test.drop(columns=["class"])
        
        return X_one_prime,y_one_prime,X_two_prime,y_two_prime,X_one_prime_test,y_one_prime_test,X_two_prime_test,y_two_prime_test
        
        
    def D_prime_data_preprocessing_splitting(self,
                                   X_one_prime,y_one_prime,X_two_prime,y_two_prime,
                                   X_one_prime_test,y_one_prime_test,X_two_prime_test,y_two_prime_test,
                                   ClassLabels
                                   ):
        
        ### Data Preprocessing

        # Target Values, Class Labels , Y 
        LabelEncoder = preprocessing.LabelEncoder()
        print("--------------------- Train ---------------------")
        
        
        y_one_prime_encode = LabelEncoder.fit_transform(y_one_prime)
        print(ClassLabels.unique())
        print("-------------------")
        print(f"Encoded class Labels of D one prime : {y_one_prime_encode}")
        # print(type(y))
        print()
        print("-------------------")
        print("-------------------")
        print()
        y_two_prime_encode = LabelEncoder.fit_transform(y_two_prime)
        print(ClassLabels.unique())
        print("-------------------")
        print(f"Encoded class Labels of D two prime : {y_two_prime_encode}")
        
        print()
        print("------------ Test Labels ------------")
        print()
        
        y_one_prime_encode_test = LabelEncoder.fit_transform(y_one_prime_test)
        print(ClassLabels.unique())
        print("-------------------")
        print(f"Encoded class Labels of D one prime test: {y_one_prime_encode_test}")
        # print(type(y))
        print()
        print("-------------------")
        print("-------------------")
        print()
        y_two_prime_encode_test = LabelEncoder.fit_transform(y_two_prime_test)
        print(ClassLabels.unique())
        print("-------------------")
        print(f"Encoded class Labels of D two prime test: {y_two_prime_encode_test}")
        
        # -------------- Train -----------------------
        # D one and two prime descritization
        d_one_prime_Encoded = self.__D_prime_descritization(X_one_prime)
        X_one_prime = d_one_prime_Encoded
        # X_one_prime.head(5)
        # df_Encoded.info()
        
        d_two_prime_Encoded = self.__D_prime_descritization(X_two_prime)
        X_two_prime = d_two_prime_Encoded
        # X_two_prime.head(5)
        
        # ----------------- Test -----------------------
        d_one_prime_Encoded_test = self.__D_prime_descritization(X_one_prime_test)
        X_one_prime_test = d_one_prime_Encoded_test
        
        d_two_prime_Encoded_test = self.__D_prime_descritization(X_two_prime_test)
        X_two_prime_test = d_two_prime_Encoded_test
        
        return X_one_prime,y_one_prime_encode,X_two_prime,y_two_prime_encode, X_one_prime_test,y_one_prime_encode_test,X_two_prime_test,y_two_prime_encode_test
        
        
    

    def __D_prime_descritization(self,data):
        # Feature vector, Instances , X 
        # encode categorical variables using label Encoder
        # select all categorical variables
    
        df_categorical = data.select_dtypes(include=['object'])
        le = preprocessing.LabelEncoder()
        df_categorical = df_categorical.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
        
        # Next, Concatenate df_categorical dataframe with original df (dataframe)
        # first, Drop earlier duplicate columns which had categorical values
    
        df_Encoded = data.drop(df_categorical.columns,axis=1)
        df_Encoded = pd.concat([df_Encoded,df_categorical],axis=1)
        return df_Encoded
    
    
        
if __name__ == "__main__":

    Adult_dataset = Classification_descisionTree()
    Adult_dataset.main()
        
        
        
        
        