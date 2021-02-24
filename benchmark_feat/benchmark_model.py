import sys
import itertools
import pandas as pd
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold, 
        cross_val_predict, train_test_split)
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline,make_pipeline
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from read_file import read_file
from convergence import convergence
import pdb
import numpy as np
import json

def mean_corrcoef(x):
    if x.shape[0] > 1000:
        x = x[np.random.choice(x.shape[0],1000),:]
    if x.shape[1] > 1000:
        x = x[:,np.random.choice(x.shape[0],1000)]

    return np.sum(np.square(np.triu(np.corrcoef(x),1)))/(len(x)*(len(x)-1)/2)
    
def benchmark_model(dataset, save_file, random_state, clf, clf_name, 
        hyper_params):

    features, labels  = read_file(dataset,label='target',
                                  classification=True)

    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True,
            random_state=random_state)

    score = roc_auc_score
    scoring = 'roc_auc'

    if len(hyper_params) > 0:
        grid_clf = GridSearchCV(clf,cv=cv, param_grid=hyper_params,
    		            verbose=3,n_jobs=1,scoring=scoring,error_score=0.0)
    else:
        grid_clf = clf
    # print ( pipeline_components)
    # print(pipeline_parameters)
    with warnings.catch_warnings():
        # Squash warning messages. Turn this off when debugging!
        # warnings.simplefilter('ignore')
        print('fitting model\n',50*'=') 
        t0 = time.process_time()
        # generate cross-validated predictions for each data point using 
        #the best estimator 
        grid_clf.fit(X_train,y_train)
        
        runtime = time.process_time() - t0
        print('done training. storing results...\n',50*'=')
        if len(hyper_params) > 0:
            best_est = grid_clf.best_estimator_
        else:
            best_est = grid_clf
      
        # store results
        print('saving output...')
        results = {}
        # get the size of the final model
        model_size=0
        num_params=0
        # pdb.set_trace()
        # get_dim() here accounts for weights in umbrella ML model
        results['num_params'] = best_est.get_n_params()+best_est.get_dim()
        results['model_size'] = best_est.get_n_nodes()
        results['model_complexity'] = best_est.get_complexity()

        # store scores
        for fold, target, X in zip(['train','test'],[y_train, y_test],
                [X_train, X_test]):
            results['roc_auc_score_'+fold] = roc_auc_score(target,
                    best_est.predict_proba(X)[:,1])
            results['ave_precision_score_'+fold] = \
                average_precision_score(target, 
                        best_est.predict_proba(X)[:,1])
       
            results['feature_space_correlation_'+fold] = \
                    mean_corrcoef(best_est.transform(X))
        params = {k:v.decode() if getattr(v, "decode", None) else v
                 for k,v in best_est.get_params().items()}

        results.update({
            'params':params,
            'random_state': random_state,
            'clf_name': clf_name,
            'version': best_est.__version__,
            'dataset': dataset.split('/')[-1].split('.')[0],
            'stats': best_est.stats,
            'archive':best_est.get_archive(justfront=True)
            })
        with open(save_file, 'w') as out:
            json.dump(results, out)

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
    parser.add_argument('-savefile', action='store', dest='SAVEFILE',default=None,
            type=str, help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
            default=None, type=int, help='Seed / trial')
    parser.add_argument('-n_jobs', action='store', dest='N_JOBS',
            default=1, type=int, help='number of cores to use')
    parser.add_argument('-dataset', action='store', dest='DATASET',
            default=None, type=str, help='endpoint name')

    args = parser.parse_args()

    # import algorithm 
    print('import from','ml.'+args.ALG)
    algorithm = importlib.__import__('ml.'+args.ALG,globals(),locals(),
                                   ['clf','name'])

    print('algorithm:',algorithm.name,algorithm.clf)
    benchmark_model(args.DATASET, args.SAVEFILE, args.RANDOM_STATE, 
                    algorithm.clf, algorithm.name, 
                    {})
