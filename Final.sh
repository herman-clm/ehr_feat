#!/bin/bash
# run comparison models
python submit_jobs.py -models GaussianNaiveBayes,DecisionTree,LogisticRegression_L1,LogisticRegression_L2,RandomForest -repeat_number 101 -n_trials 1 -folds A -results resultsFinal_r1
# run FEAT for 10 trials
python submit_jobs.py -models Feat_boolean -repeat_number 101 -n_trials 10 -folds A -results resultsFinal_r1
