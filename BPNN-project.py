# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:17:18 2023

@author: mmoein2
"""

# %%Importing Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #split data in training and testing set
from sklearn.model_selection import cross_val_score #K-fold cross validation
from sklearn.preprocessing import StandardScaler #Scaling feature
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import time
from datetime import timedelta
start_time = time.monotonic()

sns.set(style='darkgrid')


# %%Read the file
data = pd.read_csv(".... .csv", header=0)

# %%Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")


# %% Removing rows without meaningful data points
data = data[data.wfh_pre != "Question not displayed to respondent"]
data = data[data.wfh_now != "Question not displayed to respondent"]
data = data[data.wfh_expect != "Question not displayed to respondent"]
data = data[data.jobcat_now_w1b != "Question not displayed to respondent"]
data = data[data.jobcat_now_w1b != "Variable not available in datasource"]

# %% MApping data into dummies
data['wfh_pre']= data['wfh_pre'].map({'Yes':1 ,'No' :0,'':0})
data['gender']= data['gender'].map({'Female':1 ,'Male':0,'':0})
data['wfh_now']= data['wfh_now'].map({'Yes':1 ,'No':0,'':0})

# %% Introducing data variables
X = np.column_stack((data.wfh_pre, data.wfh_now, data.age, data.gender,
                     pd.get_dummies(data.educ), pd.get_dummies(data.hhincome),
                     pd.get_dummies((data.jobcat_now_w1b))))
Y = data.wfh_expect


# %% Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)#, stratify=Y)

# %% Scaling data

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
# %%
# %%
# %% Modeling
# %%
# %%


# %% Hyperparamter tunning for both activation function and number of hidden layers
Accuracy_1 = []
for j in range(0, 4):
    for i in range(1, 101):
        print(i)
        acti = ['logistic', 'tanh', 'relu', 'identity']
        # Adding solver as 'lbfgs' since it works better than the default 'adam'
        neural = MLPClassifier(activation=acti[j], solver='lbfgs', max_iter=100, hidden_layer_sizes=(i))
        
        # Cross validation
        neural_scores = cross_val_score(neural, X_train, Y_train, cv=5)
        #print("Cross Validation Accuracy: {0} (+/- {1})".format(neural_scores.mean().round(2),(neural_scores.std() * 2).round(2)))
        #print("")
        
        # Fitting final neural network
        neural.fit(X_train, Y_train)
        neural_score = neural.score(X_test, Y_test)
        Accuracy_1.append(neural_scores.mean())
        
        
# %% Plotting the performance of BPNN using each of activation function withn 100 hidden layers               
 
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(range(1, 100), Accuracy_1[1:100], color='blue')
axs[0, 0].axis('tight')
axs[0, 0].set_xlabel('Number of Hidden Layers')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_title("logistic")
axs[0, 0].set_ylim(0.6,0.8)



axs[0, 1].plot(range(1, 100), Accuracy_1[101:200], color='red')
axs[0, 1].axis('tight')
axs[0, 1].set_xlabel('Number of Hidden Layers')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].set_title("tanh")
axs[0, 1].set_ylim(0.6,0.8)


axs[1, 0].plot(range(1, 100), Accuracy_1[201:300], color='green')
axs[1, 0].axis('tight')
axs[1, 0].set_xlabel('Number of Hidden Layers')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].set_title("relu")
axs[1, 0].set_ylim(0.6,0.8)


axs[1, 1].plot(range(1, 100), Accuracy_1[301:400], color='purple')
axs[1, 1].axis('tight')
axs[1, 1].set_xlabel('Number of Hidden Layers')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].set_title("identity")
axs[1, 1].set_ylim(0.6,0.8)

plt.show()
plt.savefig('BPNN_number of hidden layers.png')

# %% Finding the best performance with lower uncertainities using each of activation function and the best number of hidden layers

Accuracy=[]
uncertanity=[]
conf=[]
for i in range(0,4):
    acti = ['logistic', 'tanh', 'relu', 'identity']
    #Adding solver as 'lbfgs' since it works better than the default 'adam'
    neural = MLPClassifier(activation=acti[i], solver='lbfgs', max_iter=100, hidden_layer_sizes=(10,))
    
    #Cross validation
    neural_scores = cross_val_score(neural, X_train, Y_train, cv=5)
    print("Cross Validation Accuracy: {0} (+/- {1})".format(neural_scores.mean().round(2), (neural_scores.std() * 2).round(2)))
    print("")
    Accuracy.append(neural_scores.mean())
    uncertanity.append((neural_scores.std() * 2).round(2))  
    #Fitting final neural network    
    neural.fit(X_train, Y_train)
    neural_score = neural.score(X_test, Y_test)
    print("Classes: {0}".format(neural.classes_))
    print("")
    print("Shape of neural network: {0}".format([coef.shape for coef in neural.coefs_]))
    print("")
    print("Coefs: ")
    print(neural.coefs_[0].round(2))
    print("")
    print(neural.coefs_[1].round(2))
    print("")
    print("Intercepts: {0}".format(neural.intercepts_))
    print("")
    print("Loss: {0}".format(neural.loss_))
    print("")
    print("Iteration: {0}".format(neural.n_iter_))
    print("")
    print("Layers: {0}".format(neural.n_layers_))
    print("")
    print("Outputs: {0}".format(neural.n_outputs_))
    print("")
    
    #Assess the fitted Neural Network
    print("Y test and predicted")
    print(Y_test.values)
    print(neural.predict(X_test))
    conf.append(neural.predict(X_test))
    print("")
    print("Mean Accuracy: {0}".format(neural_score.round(4)))
    print("")
    
# %% Plotting the confusion matrix for each of activation function in the testing phase

confusion_matrix = metrics.confusion_matrix(conf[0], Y_test,normalize='true')#, display_labels=lr.classes_, cmap="Blues", normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title('logistic')
plt.savefig('Confusion_BPNN_logistic.JPEG') #Saving the plot
plt.show()

confusion_matrix = metrics.confusion_matrix(conf[1], Y_test,normalize='true')#, display_labels=lr.classes_, cmap="Blues", normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title('tanh')
plt.savefig('Confusion_BPNN_tanh.JPEG') #Saving the plot
plt.show()

confusion_matrix = metrics.confusion_matrix(conf[2], Y_test,normalize='true')#, display_labels=lr.classes_, cmap="Blues", normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title('relu')
plt.savefig('Confusion_BPNN_relu.JPEG') #Saving the plot
plt.show()

confusion_matrix = metrics.confusion_matrix(conf[3], Y_test,normalize='true')#, display_labels=lr.classes_, cmap="Blues", normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title('identity')
plt.savefig('Confusion_BPNN_identity.JPEG') #Saving the plot
plt.show()

# %% Pltting and printing the results of the uncerntainity analysis

plt.bar(acti, Accuracy, width=0.4, yerr=uncertanity)#,color= "#659AC0") #plots the kernel function respect to accuracy and Uncertainity
plt.ylabel('Accuracy score',fontsize=14)#,fontweight='bold', family="Time")
plt.xlabel('Activation function',fontsize=14)#4,fontweight='bold', family="Time")
plt.xticks(fontsize=12)#, family="Time")
plt.ylim ((0, 1))
plt.yticks(fontsize=12)#, family="Time")
plt.savefig('BarPlot_BPNN.png',bbox_inches='tight', dpi=300)
plt.show() 
print('Mean Accuracy of Functions:')
print(Accuracy)
print("")
print('Uncertainty of Functions:')
print(uncertanity)


# %% Recording the time of runnig
time_duration=[]
end_time= time.monotonic()
time_duration.append(end_time - start_time)
print(time_duration)