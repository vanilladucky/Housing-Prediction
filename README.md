# A housing price predicting web application

## Summary
This is a data analytics and machine learning project that I undertook using a housing dataset on Kaggle in order to put my machine learning knowledge to practice and some practical application. 

## Work explained
In the data folder, there are the cleaned and external datasets.
The external data had numerical and categorical values and also numerous NaN values. I used logical imputation methods, taking into consideration the scenario, to ensure there were no NaN values and even if there were, which are logical for houses, I utilized label encoding for categorical features. 
All of this are present in the notebooks folder


Meanwhile, in the model notebook, I go onto utilize these datasets to come up with different models, varying in complexities. I went onto choose two specific algorithms which were better than the others and went onto tune their hyperparameters, and finally stacked them with linear regression for the final model, yielding the lowest error. 

## Tech used
I utilized majority of my algorithms from scikit-learn, including methods for hyperparameter tuning. Afterwards, this whole project was deployed on Streamlit (my first time utilizing this). 
I've decided to take this one step forward and deploy it as I wanted to understand what it's like a create a functioning application that can have potential users. I encountered some difficulties with virtual environments and requirements.txt. 

## Potential Areas for Improvement
It would be great if I could somehow be able to load my model's pkl or sav files so that when predicting, my model won't have to fit on the datasets which takes a considerable amount of time. My attempts have been unsuccessful as I couldn't load in a raw version of pkl files and neither could I upload online and access it from there.
If you have some advice on this, please feel free to reach out to me via email (KIMH0004@e.ntu.edu.sg)

## Web App
https://share.streamlit.io/vanilladucky/housing-prediction/main/prediction_project.py
