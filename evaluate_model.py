import pandas as pd                                                                  
import json
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

targets = {
        'htn_dx_ia':'Htndx',
        'res_htn_dx_ia':'ResHtndx', 
        'htn_hypok_dx_ia':'HtnHypoKdx', 
        'HTN_heuristic':'HtnHeuri', 
        'res_HTN_heuristic':'ResHtnHeuri',
        'hypoK_heuristic_v4':'HtnHypoKHeuri'
        }

def read_data(target, fold, repeat, data_dir):
    """Read in data, setup training and test sets"""

    drop_cols = ['UNI_ID'] + list(targets.keys())

    print('targets:',targets)
    # setup target
    
    target_new = targets[target]
    
    df_train = pd.read_csv(data_dir+'Dataset' + str(repeat) +  '/' + 
            target_new + '/' + target_new + fold + 'Train.csv')
    df_test = pd.read_csv(data_dir+'Dataset' + str(repeat) +  '/' + 
            target_new + '/' + target_new + fold + 'Test.csv')

    
    #Training
    df_y = df_train[target].astype(int)
    # setup predictors
    df_X = df_train.drop(drop_cols,axis=1)                                                    
    feature_names = df_X.columns
    print('feature names:',feature_names)
    # label encode
    print('X train info:')
    df_X.info()
    assert(not df_X.isna().any().any())
    X_train = df_X
    y_train = df_y
    
    #Testing
    df_y = df_test[target].astype(int)
    # setup predictors
    df_X = df_test.drop(drop_cols,axis=1)                                                    
    feature_names = df_X.columns
    print('feature names:',feature_names)
    # label encode

    print('X test info:')
    df_X.info()
    assert(not df_X.isna().any().any())
    X_test = df_X
    y_test = df_y

    return X_train, y_train, X_test, y_test

def evaluate_model(estimator, name, target, fold, random_state, rdir, repeat,
        data_dir='./'):
    """Evaluates estimator by training and predicting on the target."""

    X_train, y_train, X_test, y_test = read_data(target, fold, repeat, 
                                                 data_dir)
    
    # set random states
    if hasattr(estimator, 'random_state'):
        estimator.random_state = random_state
    elif hasattr(estimator, 'seed'):
        estimator.seed = random_state
    
    print('fitting to all data...')
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test.values)
    y_pred_proba = estimator.predict_proba(X_test.values)[:,1]
    
    # models.append(estimator)
    # estimator.fit(X,y)                                                 
    if type(estimator).__name__ == 'GridSearchCV':
        estimator = estimator.best_estimator_

    ### estimator-specific routines
    if 'feat' in name.lower():
        print('representation:\n',estimator.get_representation())                                  
        solution = estimator.get_representation()
        print('model:\n',estimator.get_model())
        model = estimator.get_model()
        print('archive:',estimator.get_archive(justfront=True))
        size = estimator.get_n_nodes()
        print('version:',estimator.__version__)
        print('done.')
    else:
        model = 0
        filename = rdir + '_'.join([targets[target], 
                                    name, 
                                    str(repeat), 
                                    str(random_state),
                                    str(fold), 
                                    '.pkl'])
        pickle.dump(estimator, open(filename, 'wb'))
        
        if 'LogisticRegression' in name:
            print('best C:',estimator.named_steps['est'].C_)
            size = np.count_nonzero(estimator.named_steps['est'].coef_[0])
        elif 'RandomForest' in name:
            size = 0
            for i in estimator.estimators_:
                size += i.tree_.node_count
        elif 'DecisionTree' in name:
            size = estimator.tree_.node_count
        else:
            size = len(X_train.columns)

    # get scores
    results = {}
    scorers = [accuracy_score, precision_score, average_precision_score,
               roc_auc_score]
    for (X,y,part) in zip([X_train,X_test],[y_train,y_test],['_train','_test']):
        for scorer in scorers:
            col = scorer.__name__ + part
            print(col)
            if scorer in [average_precision_score, roc_auc_score]:
                results[col] = scorer(y, 
                        estimator.predict_proba(X)[:,1])
            else:
                results[col] = scorer(y, estimator.predict(X))

    results['model'] = name
    results['target'] = targets[target]
    results['fold'] = fold
    results['RunID'] = repeat
    results['random_state'] = random_state
    results['representation'] = model
    results['size'] = size
    results['pred'] = y_pred.tolist()
    results['pred_proba'] = y_pred_proba.tolist()
    if 'feat' in name.lower():
        results['version'] = estimator.__version__
    
    
    print('results:',results)
    filename = (rdir  
                + '_'.join([targets[target], 
                            name, 
                            str(repeat),
                            str(random_state), 
                            str(fold)])
                + '.json')
    with open(filename, 'w') as out:
        json.dump(results, out, indent=4)
    
    return estimator, results

################################################################################
# main entry point
################################################################################
import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
            description="Evaluate a method on a dataset.", add_help=True)
    # parser.add_argument('-h', '--help', action='help',
    #                     help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG', default=None,
            type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-rdir', action='store', dest='RDIR',default=None,
            type=str, help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
            default=None, type=int, help='Seed / trial')
    parser.add_argument('-repeat', action='store', dest='REPEAT',
            default=1, type=int, help='repetition number')
    parser.add_argument('-fold', action='store', dest='FOLD',
            default=None, type=str, help='CV fold')
    parser.add_argument('-target', action='store', dest='TARGET',
            default=None, type=str, help='endpoint name',
            choices = ['htn_dx_ia', 'res_htn_dx_ia', 'htn_hypok_dx_ia', 
               'HTN_heuristic', 'res_HTN_heuristic', 'hypoK_heuristic_v4'])

    args = parser.parse_args()

    # import algorithm 
    print('import from','models.'+args.ALG)
    algorithm = importlib.__import__('models.'+args.ALG,globals(),locals(),
                                   ['clf','name'])

    print('algorithm:',algorithm.name,algorithm.clf)
    evaluate_model(algorithm.clf, algorithm.name, 
                   args.TARGET, args.FOLD, args.RANDOM_STATE, args.RDIR,
                   args.REPEAT)
