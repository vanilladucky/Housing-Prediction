# A housing price predicting web application

## Summary
This is a data analytics and machine learning project that I undertook using a housing dataset on Kaggle in order to put my machine learning knowledge to practice and some practical application. 

## Work explained
In the **data** folder, there are the cleaned and external datasets.
The external data had numerical and categorical values and also numerous NaN values. I used logical imputation methods, taking into consideration the scenario, to ensure there were no NaN values and even if there were, which are logical for houses, I utilized label encoding for categorical features. 


All of these data cleaning, visualization and feature engineering + categorical mapping are present in the **notebooks** folder


Meanwhile, in the **model** notebook, I go onto utilize these datasets to come up with different models, varying in complexities. I went onto choose two specific algorithms which were better than the others and went onto tune their hyperparameters, and finally stacked them with linear regression for the final model, yielding the lowest error. 

## Tech used
* Python
* Jupyter Notebook
* Scikit-Learn
* Matplotlib
* Streamlit

## Web App
https://share.streamlit.io/vanilladucky/housing-prediction/main/prediction_project.py
