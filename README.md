 # Detecting the Higgs Boson by analyzing proton collisions
In this project, we aim at solving the Higgs Boson classification, a problem posed by the CERN.  

## Folders/files 
In the this folder are the following files: 
|'run.py'| A main script containing our best method chosen to solve this classification problem : Ridge Regression
In order to run this scrpit you need to have the test.csv and train.csv in this very same folder. 

|'implementations.py'| A file containing our implementation of 6 machine learning regression and classification algorithms 
 - least squares 
 - least squares Gradient Descent
 - least squares Stochastic Gradient Descent 
 - Ridge Regression
 - Logistic regression 
 - Regularized logistic regression
 
|'proj1_helpers'| A file containing all other additional necessary functions used for 
 - loading the data
 - preprocessing
 - splitting the data
 - batch iteration 

 |'cross_validation'| A file containing all required functions for cross validation
 - build_k_indices
 - k_fold_cross_validation
 - ridge_reg_cross_validation
 - reg_log_regression_cross_validation

 
## Requirements to run the project 
The following 'Python 3' packages are necessary for running our project :
'numpy'


## Our results on AICrowd challenge 
Team name : 'Outliers'
Our team on is accessible with the following link : https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/teams/Outliers
Our best result is : 
- Categorical accuracy of 0.830
- F1 score of 0.740

## Authors
Chabenat Eugénie : eugenie.chabenat@epfl.ch
Djambazovska Sara : sara.djambazovka@epfl.ch
Mamooler Sepideh : sepideh.mamooler@epfl.ch
