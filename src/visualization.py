# Importing necessary library
import logger
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Visualize:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../log_file/visualize.txt", 'a+')
        self.data = "../data/heart.csv"
        # importing dataset
        self.data = pd.read_csv(self.data)
        # Dropping unnecessary column
        self.copy= self.data.drop(['oldpeak','slp','thall'],axis=1)
    def preprocess_and_graph(self):
        # Drawing Heatmap
        self.log_writer.log(self.file_object, "Creating Heatmap.")
        corr = self.copy.corr()
        sns.heatmap(corr)
        plt.savefig("heatmap.png")
        
        # Now we will do Uni and Bi variate analysis on our Features.
        # Countplot of Age
        self.log_writer.log(self.file_object, "Creating countplot of age.")
        plt.figure(figsize=(20, 10))
        plt.title("Age of Patients")
        plt.xlabel("Age")
        sns.countplot(x='age',data=self.copy)
        plt.savefig("countplot-age.png")
        
        # countplot of Sex
        self.log_writer.log(self.file_object, "Creating countplot of sex.")
        plt.figure(figsize=(20, 10))
        plt.title("Sex of Patients,0=Female and 1=Male")
        sns.countplot(x='sex',data=self.copy)
        plt.savefig("countplot-sex.png")
        
        # Reseting index of types of heart disease
        self.log_writer.log(self.file_object, "Reseting index of types of heart disease.")
        self.df = self.copy["cp"].value_counts().reset_index()
        self.df["index"][3] = "asymptomatic"
        self.df["index"][2] = "non-anginal"
        self.df["index"][1] = "Atyppical Anigma"
        self.df["index"][0] = "Typical Anigma"
        
        # Barplot of new indexed dataframe
        self.log_writer.log(self.file_object, "Creating Barplot of new indexed dataframe.")
        plt.figure(figsize=(20, 10))
        plt.title("Chest Pain of Patients.")
        sns.barplot(x=self.df["index"], y=self.df["cp"])
        plt.savefig("barplot-df.png")
        
        # Printing self.df
        print(self.df)
        
        # Resetting Index of  ecg report
        self.log_writer.log(self.file_object, "Resetting Index of  ecg report.")
        self.ecg = self.copy["restecg"].value_counts().reset_index()
        self.ecg["index"][0]= "normal"
        self.ecg["index"][1]= "having ST-T wave abnormality"
        self.ecg["index"][2]= "showing probable or definite left ventricular hypertrophy by Estes."
        
        print(self.ecg)
        
        # Plotting the above mention ecg dataframe
        self.log_writer.log(self.file_object, "Plotting the above mention ecg dataframe.")
        plt.figure(figsize=(20, 10))
        plt.title("ECG data of Patients")
        sns.barplot(x=self.ecg['index'],y= self.ecg['restecg'])
        plt.savefig("barplot-ecg.png")
        
        # Plotting EGG plot
        self.log_writer.log(self.file_object, "Plotting EGG plot.")
        sns.pairplot(self.copy, hue='output', data=self.copy)
        plt.savefig("EGGplot.png")
        
        # Visualizing Continous variable for trtbps and thalach
        self.log_writer.log(self.file_object, "Creating and Visualizing Continous variable for trtbps and thalach.")
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        sns.distplot(self.copy['trtbps'], kde=True, color = 'magenta')
        plt.xlabel("Resting Blood Pressure (mmHg)")
        plt.subplot(1,2,2)
        sns.distplot(self.copy['thalachh'], kde=True, color = 'teal')
        plt.xlabel("Maximum Heart Rate Achieved (bpm)")
        plt.savefig("thalach-and-trtbps-distplot.png")
        
        # Visualizing Continous variable for chol
        self.log_writer.log(self.file_object, "Creating and Visualizing Continous variable for chol.")
        plt.figure(figsize=(10,10))
        sns.distplot(self.copy['chol'], kde=True, color = 'red')
        plt.xlabel("Cholestrol")
        plt.savefig("chol-distplot.png")
        
    def std(self):
        # Since our data is not in a range so we have to do Standardisation
        self.log_writer.log(self.file_object, "Standardising..")
        self.scale=StandardScaler()
        self.scale.fit(self.copy)
        self.copy = self.scale.transform(self.copy)
            
        # Creating new dataframe
        self.log_writer.log(self.file_object, "Creating new Dataframe.")
        self.new = pd.DataFrame(self.copy,columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh','exng', 'caa', 'output'])
        print(self.new.head())
            
        # Now separating dependent and independent features
        self.log_writer.log(self.file_object, "Separating dependent and independent features")
        self.x = self.new.iloc[:,:-1]
        print(self.x)
        self.y = self.new.iloc[:,-1:]
        print(self.y)
            
        # Scikit learn train_test_split
        self.log_writer.log(self.file_object, "Performing Scikit learn train_test_split.")
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=101)
        return x_train, x_test, y_train, y_test
                
if __name__ == "__main__":
    #Visualize(data).preprocess_and_graph()
    x_train, x_test, y_train, y_test=Visualize().std()
    print(y_train)
    print(x_train)