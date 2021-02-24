import numpy as np                                                                   
from feat import Feat                                                                

clf = Feat(max_depth=6,                                                              
           max_dim = 10,
           obj='fitness,size',
           sel='lexicase',
           gens = 200,
           pop_size = 1000,
           max_stall = 20,
           stagewise_xo = True,
           scorer='log',
           verbosity=2,
           shuffle=True,
           ml='LR',
           fb=0.5,
           n_threads=1,
           classification=True,
           functions= "split,and,or,not,b2f",
           split=0.8,
           normalize=False,
           corr_delete_mutate=True, 
           simplify=0.005) 

name = 'Feat_boolean'
