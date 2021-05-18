'''
 Author      : Shiva Agrawal
 Date        : 05.09.2018
 Version     : 1.0
 Description : Multiclass classification model development using machine learning algorithm for Iris flowers data.
			   The model is used to predict the type of flower for the given new sample of features. 
'''

'''
Dataset Information:

150 samples, 4 features (all numeric), 1 output (type: string) (3 different classes): 150 x 5. 

Dataset Attribute Information

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class: 
    -- Iris Setosa
    -- Iris Versicolour
    -- Iris Virginica

	
NOTE: For this model development, I have used the template ML_modelDevelopmentTemplate.py 
      This is available in src folder of the project.
'''


#-----------------------------------------------------------------------------------------------------------------------
# step 1: Prepare Data
#-----------------------------------------------------------------------------------------------------------------------

# Load Python libraries 
#------------------------------------------------------------------

import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as pyplt
from sklearn.model_selection import train_test_split,KFold,cross_val_score

# linear models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# non linear models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# to save model 
import pickle


# b. Load Dataset from csv into Pandas Dataframe using pandas package
#-----------------------------------------------------------------------

CsvFileName = 'iris.data.csv'

# header names are extracted from iris.names.csv 
header_names = ['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm','iris_flower_class']

ML_data = pd.read_csv(CsvFileName,names = header_names)
print(ML_data.shape)
print(ML_data)

ML_data_array = ML_data.values
ML_data_input = ML_data_array[:,0:4]  # all rows and columns from index 0 to 3 (all input features)
ML_data_output = ML_data_array[:,4] # all rows and column index 4 (last column - Class (output))
print(type(ML_data_input))
print(type(ML_data_output))

#-----------------------------------------------------------------------------------------------------------------------
# step 2: Summarize Dataset
#-----------------------------------------------------------------------------------------------------------------------

# Descriptive Statistics (finding peek, mean, median, dimesions od datset, std dev, etc.)
#-----------------------------------------------------------------------------------------

print('------- Dataset Statistics--------')
print(ML_data.dtypes)  # datatypes of each input and output
print(ML_data.shape)   # Dimensions od dataset
print(ML_data.describe) # to find count, mean, median, std deviation, min value, max value, 25th and 75th percentile 
dataset_correlation = ML_data.corr()  # correlation values for each parameters
print(dataset_correlation.iloc[:,0:2])
print(ML_data.groupby('iris_flower_class').size())   # total count of samples belonging to each output class 


# b. Data Visualization (scatter matrix, histogram, density plot)
#-----------------------------------------------------------------------------------------

scatter_matrix(ML_data)
ML_data.hist()
ML_data.plot(kind = 'density', subplots = True, layout = (2,2), sharex = False)



#-----------------------------------------------------------------------------------------------------------------------
# step 3. Prepare Data
#-----------------------------------------------------------------------------------------------------------------------
# As the dataset is small and clean, no preparation is required


#-----------------------------------------------------------------------------------------------------------------------
# step 4. Evaluate Algorithms
#-----------------------------------------------------------------------------------------------------------------------

# here I have test_train split with 30 % test dataset and 70% training dataset
# then on training set, used K fold CV (k=5)
# metrics - acccuracy score
# six  ML classifications methods are used

seed = 7 
num_test = 0.3   # test/validation data - 30 %, train data - 70 %
num_splits = 5    # Cross validation using Kfold with k = 5 samples per group

# separate train and validation data
[X_train, X_validation, Y_train, Y_validation] = train_test_split(ML_data_input,ML_data_output,test_size=num_test, random_state=seed)

# intantiate different models 
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

all_result = []
all_model_names = []

for name, model in models:
    kfold = KFold(n_splits=num_splits, random_state=seed)
    results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    print('Model name: '+ name + ' Accuracy :'+ str(results.mean()*100.0) + ' ('+ str(results.std()) + ')')
    all_result.append(results)
    all_model_names.append(name)

# compare models/algorithms to find best using mean and std deviations
print(all_result)
print(all_model_names)


# Compare Algorithms to find best using visualization. Here I have used box plot for this purpose
fig = pyplt.figure()
fig.suptitle(' Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplt.boxplot(all_result)
ax.set_xticklabels(all_model_names)



# After comapring algorithms, using statistics and visualization, it is found that for the given problem LDA and SVM
# are best fit. 

#-----------------------------------------------------------------------------------------------------------------------
# 5. Improve Accuracy
#-----------------------------------------------------------------------------------------------------------------------
# This is not used here..as accuracy of the developed model is quite high



#-----------------------------------------------------------------------------------------------------------------------
# 6. Finalize Model
# a. Prediction on Validation Data set (test data set)
# b. Create Standalone Model on entire training dataset
# c. Save model for later use / convert and save model in C/C++ for production and later use
#-----------------------------------------------------------------------------------------------------------------------
# Earlier the 30 % dataset was randomly selected and kept aside for validation. Now will do validation for LDA and SVM
# using metrics

# LDA

print('-----LDA model validation--------')
lda_model_tuple = models[1]
lda_model = lda_model_tuple[1]
print(lda_model)
lda_model.fit(X_train,Y_train)
lda_predictions = lda_model.predict(X_validation)
print(accuracy_score(Y_validation, lda_predictions))
print(confusion_matrix(Y_validation, lda_predictions))
print(classification_report(Y_validation, lda_predictions))

# SVM
print('-----SVM model validation--------')
svm_model_tuple = models[5]
svm_model = svm_model_tuple[1]
svm_model.fit(X_train, Y_train)
print(svm_model)
svm_predictions = svm_model.predict(X_validation)
print(accuracy_score(Y_validation, svm_predictions))
print(confusion_matrix(Y_validation, svm_predictions))
print(classification_report(Y_validation, svm_predictions))


# After the results, it is found that LDA fits best for the given dataset.
# Hence final selected ML predictive model for iris flowers classification is LDA.


# predicting single samples just for checking (optional)
print([ML_data_input[5,:]])
print(lda_model.predict([ML_data_input[5,:]]))
print(lda_model.predict([ML_data_input[25,:]]))
print(lda_model.predict([ML_data_input[125,:]]))

# save the final model using pickle
#----------------------------------------------------------------------------------

model_filename = 'LDA_model.sav'
pickle.dump(lda_model,open(model_filename,'wb'))

# load the model and check it again whether model is correctly loaded or not (optional)
load_lda_model = pickle.load(open(model_filename,'rb'))
print('--------loaded model-------')
print([ML_data_input[5,:]])
print(load_lda_model.predict([ML_data_input[5,:]]))
print(load_lda_model.predict([ML_data_input[25,:]]))
print(load_lda_model.predict([ML_data_input[125,:]]))


pyplt.show()