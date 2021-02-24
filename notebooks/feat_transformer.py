"""Model features
(reversed)
[(sum_enc_during_htn_meds_3>=1.500000)]
[(median_enc_during_htn_meds_4_plus>=1.250000)]
[sd_enc_during_htn_meds_2]
[(mean_systolic>=128.641357)]
[(max.CALCIUM>=10.150000)]
[(re_htn_spec_sum>=40.500000)]
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

class FeatTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.ss = StandardScaler()
        self.feature_names = [
            'sum_enc_during_htn_meds_3>1',
            'median_enc_during_htn_meds_4_plus>1.25',
            'sd_enc_during_htn_meds_2',
            'mean_systolic>128.6',
            'max.CALCIUM>10.1',
            're_htn_spec_sum>40']
    def fit(self, X, y=None):
        self.ss.fit(self.feat_model(X))
        return self
    
    def feat_model(self, X):
        if type(X).__name__ == 'DataFrame':
            X = X.values
        Phi = []
        Phi.append(X[:,280] >= 1.5)
        Phi.append(X[:,285] >= 1.25)
        Phi.append(X[:,287])
        Phi.append(X[:,19]>=128.641357)
        Phi.append(X[:,89]>=10.15)
        Phi.append(X[:,308]>=40.5)
       
        Phi = np.array(Phi).transpose()
        return Phi
    
    def transform(self, X):
        Phi = self.ss.transform(self.feat_model(X))
        return Phi
