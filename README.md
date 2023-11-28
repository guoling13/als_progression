# als_progression
Code for the paper "Predicting Amyotrophic Lateral Sclerosis (ALS) Progression with Machine Learning" published in ALS-FTD journal. 

# Usage
1. clean_and_merge_data.ipynb : reads, clean and merge all the raw data from the PROACT database.
2. impute_and_window.ipynb : further cleans the data, split the data into 5-folds for cross validation, does data imputation and finally window the data into different observation and prediction windows.
3. fast_nonfast_xgboost.py : does XGBoost classification, hyperparameter tuning and evaluation for all the different observation and prediction window lengths, and save the results.
