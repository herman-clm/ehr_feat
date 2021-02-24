"""Model selection strategies for FEAT's final population"""
import pandas as pd

def smallest_of_best_half(data):
    data = data.nlargest(5, columns=['average_precision_score_train'])
    return data.nsmallest(1, columns=['size'])
    
def smallest_of_best_quartile(data):
    print('taking models with APS >=best_quartile:',
        data['average_precision_score_train'].quantile(0.75),'...')
    data = data.loc[data['average_precision_score_train'] >= data['average_precision_score_train'].quantile(0.75)]
    return data.nsmallest(1, columns=['size'])

def smallest_of_best_three_quartiles(data):
    print('taking models with APS >= lowest_quartile:',
        data['average_precision_score_train'].quantile(0.25),'...')
    data = data.loc[data['average_precision_score_train'] >= data['average_precision_score_train'].quantile(0.25)]
    return data.nsmallest(1, columns=['size'])

def best_of_smallest_three_quartiles(data):
    print('taking models with size <= smallest 3 quartiles:',
        data['size'].quantile(0.25),'...')
    data = data.loc[data['size'] <= data['size'].quantile(0.75)]
    return data.nlargest(1, columns=['average_precision_score_train'])

def best_of_smallest_half(data):
    data = data.nsmallest(5, columns=['size'])
    return data.nlargest(1, columns=['average_precision_score_train'])

def best_of_smallest_quartile(data):
    print('taking models with smaller than lowest_quartile:',
        data['size'].quantile(0.25),'...')
    data = data.loc[data['size'] <= data['size'].quantile(0.25)]
    return data.nlargest(1, columns=['average_precision_score_train'])

def best(data):
    return data.nlargest(1, columns=['average_precision_score_train'])
    
def smallest(data):
    return data.nsmallest(1, columns=['size'])

def select_feat_models(df, method = smallest_of_best_half):
   #FEAT 10x rerun and selection procedure: 
    frames = [] 
    for (run_id, fold, target), dfg in df.groupby(['RunID','fold','target']):
#         print('RunID:',run_id,'fold:',fold)
        data = dfg.reset_index(drop=True)
        data = method(data)
        frames.append(data)
    return pd.concat(frames)
