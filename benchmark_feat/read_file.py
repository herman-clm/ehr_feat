import pandas as pd
import numpy as np
import pdb
from sklearn.preprocessing import RobustScaler, LabelEncoder

def read_file_htn(filename):
    targets = {
            'htn_dx_ia':'Htndx',
            'res_htn_dx_ia':'ResHtndx', 
            'htn_hypok_dx_ia':'HtnHypoKdx', 
            'HTN_heuristic':'HtnHeuri', 
            'res_HTN_heuristic':'ResHtnHeuri',
            'hypoK_heuristic_v4':'HtnHypoKHeuri'
            }
    rev_targets = {v:k for k,v in targets.items()}

    drop_cols = ['record_id','redcap_data_access_group','htn_dx_ia',
            'res_htn_dx_ia', 'htn_hypok_dx_ia','RACE_MASTER_CODE',
            'GENDER_MASTER_CODE','ZIP_CAT', 'SERVICE_MASTER_DESCRIPTION',
            'MASTER_LOCATION_CODE'] + list(targets.keys())

    
    # setup target
    target_new = filename.split('/')[-1].split('Train')[0][:-1]
    target= rev_targets[target_new]
    
    df_train = pd.read_csv(filename)
    
    #Training
    df_y = df_train[target].astype(int)
    # setup predictors
    df_X = df_train.drop(drop_cols,axis=1, errors='ignore')

    return df_X, df_y.values

def read_file(filename, classification=True, label='class', sep=None):
    
    if 'Htn' in filename:
        return read_file_htn(filename)
    
    if filename.split('.')[-1] == 'gz':
        compression = 'gzip'
    else:
        compression = None

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression,
                engine='python')
    
    X = input_data.drop(label, axis=1)
    y = input_data[label].values

    # X = RobustScaler().fit_transform(X)

    # if classes aren't labelled sequentially, fix
    if classification:
        y = LabelEncoder().fit_transform(y)

    return X, y 


