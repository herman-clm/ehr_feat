from feat import Feat                                                                
from .common_args import common_args
common_args.update({
    'simplify':True
    })

clf = Feat(**common_args)

name = 'Feat_simplify'
