# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains a bank-marketing campaign with employees, level of education, age, job, marital status and column "y". We want to predict they column "y" that has 0 or 1 values using classification algorithms.


The best performing model was the VotingEnsemble using the AutoML experiment.


## Scikit-learn Pipeline
First we imported the dataset from an URL address, made same changes in the dataset using the function clean_data where we changed some columns as month, weekdays using the one_hot enconding method. 
After that, we divided the dataset in train and test.
We tunned the Regularization Strength and the Max Iterations. 
We used the Logistic Regression algorithm.

In the paremeter sample, we changed the value of C and max_iter.
Lower values of C cause stronger regularization.
And max_iter specifies the maximum number of iterations to converge.


We used the BanditPolicy.
The frecuency for applying the policy was 3.
The slack factor is the ratio used to calculate the allowed distance from the best performing experiment run.

## AutoML
With the AutoML the best model was the VotingEnsemble with the higher accuracy among the other models.

## Pipeline comparison
The Logistic Regression model obtained a lower accuracy than the Voting Ensemble model.
In the Voting Ensemble model, the duration column was the main factor for explaning the model.

## Future work
We could improve with a larger database, changeing the parameter sample and the early termination policy. 


