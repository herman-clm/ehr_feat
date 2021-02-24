from glob import glob                                                           
import os                                                                       
import sys                                                                      
import argparse                                                                 
from joblib import Parallel, delayed
                                                                                
if __name__ == '__main__':                                                      
    parser = argparse.ArgumentParser(description="Submit long jobs.",           
                                     add_help=True)                            
    parser.add_argument('-target',action='store',dest='TARGET', type=str,              
            default=('htn_dx_ia,'
                     'res_htn_dx_ia,'
                     'htn_hypok_dx_ia,'
                     'HTN_heuristic,'
                     'res_HTN_heuristic,'
                     'hypoK_heuristic_v4'))
    parser.add_argument('-folds',action='store',dest='FOLDS', type=str,              
            default='A,B,C,D,E')
    parser.add_argument('-repeats',action='store',dest='REPEATS', type=int,              
            default=1)
    parser.add_argument('-repeat_number',action='store',dest='REPEAT_N', 
            type=int, default=-1)
    parser.add_argument('-models',action='store',dest='MODELS', type=str,              
            default=('Feat_boolean,'
                     'DecisionTree,'
                     'Feat_init_l2,'
                     'Feat_init_l1,'
                     'RandomForest,'
                     'GaussianNaiveBayes,'
                     'LogisticRegression_L1,'
                     'LogisticRegression_L2'
                     ))                  
    parser.add_argument('--long',action='store_true',dest='LONG', default=False)
    parser.add_argument('-seeds',action='store',type=str,dest='SEEDS',          
            default='14724,24284,31658,6933,1318,16695,27690,8233,24481,6832,'  
            '13352,4866,12669,12092,15860,19863,6654,10197,29756,14289,'        
            '4719,12498,29198,10132,28699,32400,18313,26311,9540,20300,'        
            '6126,5740,20404,9675,22727,25349,9296,22571,2917,21353,'           
            '871,21924,30132,10102,29759,8653,18998,7376,9271,9292')            
    parser.add_argument('-results',action='store',dest='RDIR',                  
            default='results/',type=str,help='Results directory')               
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,  
            type=int, help='Number of trials to run')                           
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=1,          
            type=int, help='Number of parallel threads per job')                           
    parser.add_argument('-m',action='store',dest='M',default=8000,type=int,        
                        help='LSF memory request and limit (MB)')               
    parser.add_argument('--local', action='store_false', dest='LSF',               
            default=True, help='Run locally instead of submitting jobs')         
    parser.add_argument('-q',action='store',dest='QUEUE',
            default='epistasis_long', type=str, help='LSF queue name')               
    args = parser.parse_args()                                                  
                                                                                
    n_trials = len(args.SEEDS) if args.N_TRIALS < 1 else args.N_TRIALS
    
    print('n_trials: ', n_trials)
    seeds = args.SEEDS.split(',')[:n_trials]                          
    print('using these seeds:',seeds)
    print('for folds:',args.FOLDS)
    folds = args.FOLDS.split(',')
    print('for # of repeats:',args.REPEATS)
    if args.REPEAT_N != -1:
        repeats = [args.REPEAT_N]
        if args.REPEATS != 1:
            raise ValueError('if -repeat_number is specified, '
                    '-repeats must be 1')
    else:
        repeats = range(args.REPEATS)
    print('and these targets:',args.TARGET)                                        
    if args.LONG:                                                               
        q = 'epistasis_long,mooreai_long'                                       
    else:                                                                       
        # q = 'epistasis_normal,mooreai_normal'
        q = 'epistasis_normal'                                                  
    if not args.LSF:
        lpc_options = ''  
    else:
        lpc_options = '--lsf -q {Q} -m {M} -n_jobs {NJ}'.format(
            Q=q,                                                                
            M=args.M,                                                           
            NJ=args.N_JOBS)                                                     
                                                                                

    all_commands = []
    job_info = [] 
    
    for repeat in repeats:
        for target in args.TARGET.split(','):
            for ml in args.MODELS.split(','):
                filepath = '/'.join([args.RDIR,target,ml]) + '/'
                if not os.path.exists(filepath):
                    print('WARNING: creating path',filepath)
                    os.makedirs(filepath)
                for seed in seeds:
                    for fold in folds:
                        random_state = seed
                        all_commands.append('python evaluate_model.py '
                                    ' -ml {ML}'
                                    ' -target {TARGET}'
                                    ' -seed {RS}'
                                    ' -rdir {RDIR}'
                                    ' -fold {FO}'
                                    ' -repeat {REPEAT}'.format(
                                                           ML=ml,
                                                           TARGET=target,
                                                           RS=random_state,
                                                           RDIR=filepath,
                                                           FO=fold,
                                                           REPEAT=repeat,
                                                           NJ=args.N_JOBS
                                                          )
                                            )                   
                        job_info.append({
                            'ml':ml,
                            'target':target,
                            'fold':fold,
                            'repeat':repeat,
                            'results_path':filepath,
                            'seed':random_state
                            })

    if args.LSF:    # bsub commands
        for i,run_cmd in enumerate(all_commands):
            job_name = '_'.join([str(s) for s in [
                job_info[i]['target'], 
                job_info[i]['ml'],
                job_info[i]['seed'],
                job_info[i]['fold'],
                job_info[i]['repeat']]
                ])
            out_file = job_info[i]['results_path'] +'/'+ job_name + '_%J.out'
            error_file = out_file[:-4] + '.err'

            # choose uniformly among queues if more than one available
            if ',' in args.QUEUE:
                queue = np.random.choice(args.QUEUE.split(','))
            else: 
                queue = args.QUEUE
            
            bsub_cmd = ('bsub -o {OUT_FILE} -n {N_CORES} -J {JOB_NAME} '
                    '-q {QUEUE} -R "span[hosts=1] rusage[mem={M}]" -M {MMAX} '
                    '').format(
                               OUT_FILE=out_file,
                               JOB_NAME=job_name,
                               QUEUE=queue,
                               N_CORES=args.N_JOBS,
                               M=args.M,
                               MMAX=32000)
            
            bsub_cmd +=  '"' + run_cmd + '"'
            print(bsub_cmd)
            os.system(bsub_cmd)     # submit jobs 

    else:   # run locally  
        for run_cmd in all_commands: 
            print(run_cmd) 
        Parallel(n_jobs=args.N_JOBS)(delayed(os.system)(run_cmd) 
                for run_cmd in all_commands )
