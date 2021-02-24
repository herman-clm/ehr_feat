from glob import glob
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit long jobs.",
                                     add_help=False)
    parser.add_argument('DATA_PATH',type=str)
    parser.add_argument('-ml',action='store',dest='mls', type=str, 
            default='Feat_base,'
                    'Feat_onedim,'
                    # 'Feat_corr_delete_mutate,'
                    'Feat_simplify001,'
                    # 'Feat_simplify005,'
                    'Feat_simplify010,'
                    'Feat_simplify100,'
                    'Feat_boolean,'
                    'Feat_boolean_simplify001,'
                    # 'Feat_boolean_simplify005,'
                    'Feat_boolean_simplify010,'
                    # 'Feat_boolean_simplify050,'
                    'Feat_boolean_simplify100'
                    # 'Feat_boolean_simplify500'
                )
    parser.add_argument('-results',action='store',dest='RDIR',type=str,
            default='results')
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',type=int,
            default=1)

    args = parser.parse_args()

    datapath = args.DATA_PATH 

    if not os.path.exists(args.RDIR):
        os.mkdir(args.RDIR)

    lpc_options = '--lsf -q epistasis_normal -m 8000 -n_jobs 1'

    closest_datasets = \
    ['clean1', 'clean2', 'Hill_Valley_with_noise', 'Hill_Valley_without_noise',
    'sonar', 'molecular-biology_promoters', 'spambase', 'tokyo1', 'spectf',
    'ionosphere', 'coil2000', 'chess', 'kr-vs-kp', 'backache',
    'breast-cancer-wisconsin', 'wdbc', 'dis', 'hypothyroid', 'colic', 
    'horse-colic']

    for cd in closest_datasets:
        f = datapath + '/classification/' + cd + '/' + cd + '.tsv.gz'
        jobline =  ('python analyze.py {DATA} '
               '-ml {ML} '
               '-results {RDIR} -n_trials {NT} {LPC}').format(
                       RDIR=args.RDIR,
                       NT=args.N_TRIALS,
                       DATA=f,
                       LPC=lpc_options,
                       ML=args.mls)
        print(jobline)
        os.system(jobline)

    # submit hypertension jobs
    for i,f in enumerate(glob('../dataset/*/*Train.csv')):
        jobline =  ('python analyze.py {DATA} '
               '-ml {ML} '
               '-results {RDIR} -n_trials {NT} {LPC}').format(
                       RDIR=args.RDIR,
                       NT=args.N_TRIALS,
                       DATA=f,
                       LPC=lpc_options,
                       ML=args.mls)
        print(jobline)
        os.system(jobline)
