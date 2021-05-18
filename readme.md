# Multi Class Classification Predictive Model development for Iris flowers data

This project use the datset of Iris flowers and develop the Machine learning predictive model (multi class classification). It first import the dataset into Pandas Dataframe structure and then do the prepreprocessing and analysis of the data using scikit-learn library of Python.

After that, multiple ML models (using scikit-learn) are developed both linear and non linear for the same dataset with 30 % validation data and 70 % training data. All the developed models are then compared to find the best using Kfold cross validation and classification accuracy metric.

For this, it is found that LDA (Linear Discriminant Analysis) and SVM (Support Vector Machine) classifier fits the best. 
Hence both these models are further investigated and finally LDA is selected for the model.

The developed model is then used to predict output using some random samples from the complete dataset. The model is then saved using the Pickle library of the Pyhton for future use.

The project is implemented in Python (with Pycharm IDE) and Ubuntu 16.04 OS.

## Folder structure

1. docs
    * project details.pdf                                 - it is short project report
    * ML_modelDevelopmentTemplate.py                      - template used as reference to follow the steps for model development
2. results
    * LDA_model.sav                                       - developed model saved as .sav file
    * results.txt                                         - copy of the console output of the src code of the project
3. src
    * dataset (folder)                                    - it contains the data and header of the Iris flower dataset as two csv files
    * Multiclass Classification Model Development.py      - documented and tested source code of the project
                                

