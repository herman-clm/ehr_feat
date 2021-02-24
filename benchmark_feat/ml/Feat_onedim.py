from feat import Feat                                                                
from .common_args import common_args
common_args.update({
    'max_dim':1
    })
clf = Feat(**common_args) 

name = 'Feat_1dim'
