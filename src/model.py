import logger
import pickle
import numpy as np
import matplotlib.pyplot as plt
from visualization import Visualize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold,KFold

class model_selection:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("../log_file/model_selectionlog.txt", 'a+')
        self.x_train,self.x_test,self.y_train,self.y_test = Visualize().std()
        
    """def encoder(self):
        self.lbe = LabelEncoder()
        self.encoded_y = self.lbe.fit_transform(self.y_train)
        self.encoded_y_test = self.lbe.fit_transform(self.y_test)
        
        #return self.encoded_y
    """
    def knn(self):
        self.error_rate= []
        for i in range(1,40):
            knn= KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.x_train,self.encoded_y)
            pred= knn.predict(self.x_test)
            self.error_rate.append(np.mean(pred != self.encoded_ytest))
            
            plt.figure(figsize=(10,6))
            plt.plot(range(1,40),self.error_rate,color='blue', linestyle='dashed', marker='o',
                    markerfacecolor='red', markersize=10)
            plt.xlabel('K Vlaue')
            plt.ylabel('Error rate')
            plt.title('To check the correct value of k')
            plt.savefig("KNN_Knee.png")
            #plt.show()
    def logreg(self):
        self.lbe = LabelEncoder()
        self.encoded_y = self.lbe.fit_transform(self.y_train)
        self.encoded_y_test = self.lbe.fit_transform(self.y_test)
        
        self.logreg = LogisticRegression()
        model = self.logreg.fit(self.x_train, self.encoded_y)
        self.y_pred1 = model.predict(self.x_test)
        
        # saving Model
        pickle.dump(model, open('model.pkl', 'wb'))

        #print(y_pred1)
        self.log_writer.log(self.file_object, "Calculating accuracy Score.")
        self.logreg_acc_score = accuracy_score(self.encoded_y_test, self.y_pred1)
        print(self.logreg_acc_score*100,"%")

        self.log_writer.log(self.file_object, "Doing Hyperparameter tuning.")
        param_grid = {"penalty": ["none", "l2"],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "solver": ["newton-cg", "lbfgs", "sag", "saga"]}
        #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        cv = KFold(n_splits=10,shuffle=True,random_state=1)
        grid_search = GridSearchCV(estimator=self.logreg, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(self.x_train, self.encoded_y)

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            #print("%f (%f) with: %r" % (mean, stdev, param))
            self.log_writer.log(self.file_object,"%f (%f) with: %r" % (mean, stdev, param))
        self.log_writer.log(self.file_object, "Finish Hyperparameter tuning for logistic regression using KFold.")   
    
    def knn(self):
        self.lbe = LabelEncoder()
        self.encoded_y = self.lbe.fit_transform(self.y_train)
        self.encoded_y_test = self.lbe.fit_transform(self.y_test)
        
        self.log_writer.log(self.file_object, "Fitting KNN model.")
        self.knn= KNeighborsClassifier(n_neighbors=12)
        self.knn.fit(self.x_train,self.encoded_y)
        self.y_pred2= self.knn.predict(self.x_test)
        
        self.log_writer.log(self.file_object, "Calculating accuracy Score.")
        self.knn_acc_score = accuracy_score(self.encoded_y_test, self.y_pred2)
        print(self.knn_acc_score*100,"%")
        self.log_writer.log(self.file_object, "self.knn_acc_score*100,'%'")
        self.log_writer.log(self.file_object, "Finished Fitting KNN model.")
        
        self.log_writer.log(self.file_object, "Starting Hyperparameter tuning KNN Begins,")
        n_neighbors = range(1, 21, 2)
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']
        grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=self.knn, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(self.x_train,self.encoded_y)
        print(grid_search.best_params_)
        self.log_writer.log(self.file_object,"grid_search.best_params_")
        
        self.log_writer.log(self.file_object, "Fitting KNN model with tuned parameter.")
        knn= KNeighborsClassifier(n_neighbors=12,metric='manhattan',weights='distance')
        knn.fit(self.x_train,self.encoded_y)
        knn_pred= knn.predict(self.x_test)
        
        knn_pred_acc_score = accuracy_score(self.encoded_y_test, knn_pred)
        print(knn_pred_acc_score*100,"%")
        
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            #print("%f (%f) with: %r" % (mean, stdev, param))
            self.log_writer.log(self.file_object,"%f (%f) with: %r" % (mean, stdev, param))
        self.log_writer.log(self.file_object, "Finish Hyperparameter tuning for KNN.")
        self.log_writer.log(self.file_object, "Finished KNN Training.")
    
    def svm(self):
        self.lbe = LabelEncoder()
        self.encoded_y = self.lbe.fit_transform(self.y_train)
        self.encoded_y_test = self.lbe.fit_transform(self.y_test)
        
        self.log_writer.log(self.file_object, "Fitting svm model.")
        self.svm = svm.SVC()
        self.svm.fit(self.x_train,self.encoded_y)
        self.y_pred3= self.svm.predict(self.x_test)
        svm_acc_score = accuracy_score(self.encoded_y_test, self.y_pred3)
        print(svm_acc_score*100,"%")
        self.log_writer.log(self.file_object, "svm_acc_score*100,'%'")
        
        self.log_writer.log(self.file_object,"Started Tuning SVM.")
        kernel = ['poly', 'rbf', 'sigmoid']
        C = [50, 10, 1.0, 0.1, 0.01]
        gamma = ['scale']
        
        grid = dict(kernel=kernel,C=C,gamma=gamma)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=self.svm, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        
        grid_result = grid_search.fit(self.x_train,self.encoded_y)
        print(grid_search.best_params_)
        
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            #print("%f (%f) with: %r" % (mean, stdev, param))
            self.log_writer.log(self.file_object,"%f (%f) with: %r" % (mean, stdev, param))
        self.log_writer.log(self.file_object, "Finish Hyperparameter tuning for SVM.")
        
        
    
if __name__ == "__main__":
    modell = model_selection()
    modell.logreg()
        
        
