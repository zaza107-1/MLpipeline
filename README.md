# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains a bank-marketing campaign with employees, level of education, age, job, marital status and column "y". We want to predict they column "y" that has 0 or 1 values using classification algorithms.


The best performing model was the VotingEnsemble using the AutoML experiment.

## Scikit-learn Pipeline
Within the pipeline, we imported a dataset from a URL. Then we applied the train test split with a 33% of the data used as test.
The classification technique used was the Logistic Regression.
We used the Hyperdrive function to carry out the hyperparameter optimization. We used the Accuray metric for achieving the goal of the Hyperdrive.

As we mentioned before, first we imported the dataset from an URL address, made same changes in the dataset using the function clean_data where we changed some columns as month, weekdays using the one_hot enconding method. 
After that, we divided the dataset in train and test.
We tunned the Regularization Strength and the Max Iterations with the Hyperdrive method. 
We used the Logistic Regression algorithm.

In the paremeter sample, we changed the value of C and max_iter.
Lower values of C cause stronger regularization.
And max_iter specifies the maximum number of iterations to converge.
The benefits of the choosen parameters sample are to reduce the computational costs and speed up the results reducing the max_iter for example. 

We used the BanditPolicy.
The frecuency for applying the policy was 3.
The slack factor is the ratio used to calculate the allowed distance from the best performing experiment run.
The benefits of the early termination policy is also for reducing the time spent trying to look for improvments in the model. When the improvments of the results aren't significantly better, we stop the process.

## AutoML
The objective of the AutoML was to detect the best model that reachs the higger value of the Accuracy metric.
With the AutoML the best model was the VotingEnsemble with the higher accuracy among the other models.
In the Voting Ensemble model, the duration column was the main factor for explaning the model.
Voting Ensemble improves the machine learning performance by combining multiple models. Their predictions are based on the weighted average of predicted class probabilities (for classification tasks) according to the documentation.
We can see the parameters of the prefittedsoftvotingclassifier with max_iter equal to 1000. n_jobs equal to 1. Learning rate 'constant'. L1_ratio = 0.8367 and power_t = 0.2222. Penalty = none. random_state = 1.

## Pipeline comparison
The Logistic Regression model obtained a lower accuracy than the Voting Ensemble model.
With the Hyperdrive method we got an Accuracy of 0.9101 and, with the AutoML model we got an Accuracy of 0.9162 whe we can see that the AutoML got a higher accuracy. But the difference between them is small.
But the best thing of the AutoML is that everything is done automatically, testing different algorithms as to get the highest metric that we choose. In this case we choose the Accuracy metric, but we can change to AUC, RSE or anything else.

## Future work
We could improve the Accuracy of the models with a larger database, changing the parameter sample and the early termination policy. Or we can add more columns doing some engineering using the columns that we have now.


