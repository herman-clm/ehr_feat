from feat import Feat                                                                
from .common_args import common_args

clf = Feat(**common_args,
           ################################
           functions="split,split_c,and,or,not,b2f,c2f",
           simplify=True
           ################################
           ) 

name = 'Feat_boolean_simplify'
