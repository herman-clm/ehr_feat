#!/bin/bash
python submit_jobs.py -repeat 50 -models GaussianNaiveBayes,DecisionTree,LogisticRegression_L1,LogisticRegression_L2,RandomForest  -n_trials 1
python submit_jobs.py -repeat 50 -models Feat_boolean  -n_trials 10
