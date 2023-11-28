import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report, average_precision_score
from sklearn.model_selection import KFold


prefix = 'fast_nonfast_data/fast_nonfast_'

def run(X_train, dtrain, dtest):
    # start with default params
    params = {'objective':'binary:logistic', 'scale_pos_weight':sum(y_train==0)/sum(y_train), 'eval_metric':['logloss','aucpr','auc'], 'seed':42, # not tuned
               'max_depth': 6,
               'alpha': 0, 'reg_lambda':1, 'eta': 0.3, 
               'colsample_bytree': 1, 'subsample':1,  
              }
    
    # split train set into 5 folds for CV
    patients = X_train['subject_id'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    folds = []
    for train_index, test_index in kf.split(patients):
        folds.append((X_train[X_train['subject_id'].isin(patients[train_index])].index.tolist(),X_train[X_train['subject_id'].isin(patients[test_index])].index.tolist()))

    # tune subsample, regularization, max depth, eta accordingly
    # example hyperparameters for single visit observation, 6 months prediction window
    best_params = {'objective': 'binary:logistic', 'scale_pos_weight': 2.8222902633190445, 'eval_metric': ['logloss', 'aucpr', 'auc'], 'seed': 42, 'max_depth': 2, 'alpha': 100, 'reg_lambda': 500, 'eta': 0.2, 'colsample_bytree': 1.0, 'subsample': 0.9}
    best_boost = 214

    print(best_params)
    print(best_boost)

    # train to best iteration
    evallist = [(dtrain, 'train'), (dtest, 'eval')] 
    xg = xgb.train(dtrain=dtrain, params=best_params, evals=evallist, num_boost_round=best_boost+1)
    
    return xg, best_params


def evaluate(model, dtrain, y_train, dtest, y_test):
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # evaluate on training data
    train_preds = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))
    fpr, tpr, thres = roc_curve(y_train, train_preds) # use train data to set threshold
    threshold = thres[np.argmax(tpr-fpr)]
    train_auc = roc_auc_score(y_train, train_preds)
    train_auprc = average_precision_score(y_train, train_preds)

    # evaluate on testing data
    preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    
    # AUC
    auc = roc_auc_score(y_test, preds)
    print('AUC: {}, best threshold: {}'.format(auc, threshold))

    # Area under precision recall curve
    auprc = average_precision_score(y_test, preds)
    print('AUPRC: {}'.format(auprc))

    res = classification_report(y_test, (preds>=threshold), output_dict=True)
    print(classification_report(y_test, (preds>=threshold), digits=3))
    print(confusion_matrix(y_test, (preds>=threshold)))


    return {'train_auc':train_auc, 'train_auprc':train_auprc,
            'auc':auc, 'auprc':auprc, 'threshold':threshold, 'acc':res['accuracy'], 'sens':res['1']['recall'],
            'spec':res['0']['recall'], 'f1':res['1']['f1-score'], 'ppv':res['1']['precision'], 'npv':res['0']['precision'],
            'confusion_mat':confusion_matrix(y_test, (preds>=threshold)).tolist()
            }, preds


# observation and prediction windows
obs_wins = [0,3,6,12]

# features
drop_cols = ['subject_id', 'delta',
            'n_obs_visits', 'n_pred_visits','y_mean','y_sd','y_slope', 'fast', 'train']
pred_wins = [3,6,12]

# initialize outputs
feats = pd.read_csv(prefix + '2mos_12mos_cv0.csv', index_col=[0])
feat_importance = pd.DataFrame({'count':0, 'gain':0}, index=feats.drop(columns=drop_cols).columns)
res = pd.DataFrame()

for cv in range(5):
    for o in obs_wins:
        for p in pred_wins:
            print('CV: {}, Obs win: {}, Pred win: {}'.format(cv, o, p))
            
            df_win = pd.read_csv(prefix + str(o) + 'mos_' + str(p) + 'mos_cv' + str(cv) + '.csv', index_col=[0])
            
            X_train = df_win[df_win['train']==1]
            X_train = X_train.reset_index(drop=True)
            
            X_test = df_win[df_win['train']==0]
            X_test = X_test.reset_index(drop=True)

            y_train = X_train['fast']
            y_test = X_test['fast']
            
            dtrain = xgb.DMatrix(data=X_train.drop(columns=drop_cols),label=y_train, missing=np.nan)
            dtest = xgb.DMatrix(data=X_test.drop(columns=drop_cols),label=y_test, missing=np.nan)

            xg, best_params = run(X_train, dtrain, dtest)
            
            # update feature importance
            scores = xg.get_score(importance_type='gain')
            scores = pd.DataFrame([scores[key] for key in scores], index=scores.keys(), columns=['_'.join((str(o),str(p),str(cv)))])
            feat_importance = feat_importance.join(scores)
            
            # evaluate model
            results, preds = evaluate(xg, dtrain, y_train, dtest, y_test)

            # save predictions
            preds = pd.DataFrame(preds)
            preds.to_csv(prefix + '{}mos_{}mos_cv{}_pred.csv'.format(o,p,cv))
            
            # Update and save results
            res_update = {'cv':cv, 'obs':o, 'pred':p, 'hyperparameters':best_params, 'num_boost_rounds':xg.best_iteration+1,
                            'train_size':len(X_train), 'test_size':len(X_test), 
                            'label_perc_train':sum(y_train)/len(y_train), 'label_perc_test':sum(y_test)/len(y_test),
                            'label_perc':sum(df_win['fast'])/len(df_win)
                            }
            res_update.update(results)
            res = res.append(res_update, ignore_index=True)

            # save results
            res.to_csv(prefix + 'results.csv')
            feat_importance.to_csv(prefix + 'feat_importance.csv')
