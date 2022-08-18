# Using FEAT to learn computable phenotypes

This repo contains the code to reproduce the results obtained in the following paper:
La~Cava, W., Lee, P. C., Ajmal, I., Ding, X., Solanki, P., Cohen, J. B., Moore, J. H., & Herman, D. S. (2020). 
Application of concise machine learning to construct accurate and interpretable EHR computable phenotypes. 
[MedRxiv, 2020.12.12.20248005.](https://doi.org/10.1101/2020.12.12.20248005)

- For the results on MIMIC-III, see https://github.com/cavalab/mimic3-benchmarks. 


## How to Use
Experiments are submitted via the `submit_jobs.py` function. 
Run `python submit_jobs.py -h` to see options. 

## Experiments

### Benchmark FEAT changes

These experiments are in the `benchmark_feat` folder.
To run them, navigate there and run 

`python submit_jobs.py`

For configuration options, see `python submit_jobs.py -h`.

### Predicting Heuristics and Phenotypes

Run `bash Alltest.sh` for five-fold cross-validation over 50 iterations (all results saved in results/).
Run `bash Final.sh` for training the final model (all results in saved in resultsFinal/).
See notebooks/ for model selection, generating figures, etc. 

## Dependencies

- Python 3+
- Anaconda or Miniconda
- [Feature Engineering Automation Tool](github.com/lacava/feat)

## Installing FEAT

Use the [environment.yml](environment.yml) file to install Feat. 
Installation steps are provided in `feat_setup.sh`, and also shown below:

```bash
git clone https://github.com/lacava/feat # clone the repo
cd feat # enter the directory
conda env create -f environment.yml
conda activate feat-env
#add some environment variables
export SHOGUN_LIB=/path/to/anaconda/envs/feat-env/lib/
export SHOGUN_DIR=/path/to/anaconda/envs/feat-env/include/
export EIGEN3_INCLUDE_DIR=/path/to/anaconda/envs/feat-env/include/eigen3/
# install feat
./configure  
./install y
```

## Contact

 - William La Cava: lacava@upenn.edu
 - Paul Lee: crlee@sas.upenn.edu
 - Daniel Herman: Daniel.Herman2@pennmedicine.upenn.edu

## Acknowledgments

We would like to thank Debbie Cohen for helpful discussions about secondary hypertension. 
W. La Cava was supported by NIH grant K99-LM012926.
This work was supported by Grant 2019084 from the Doris Duke Charitable Foundation and the University of Pennsylvania. 
W.La Cava was supported by NIH grant K99LM012926. 
J.H. Moore and W. La Cava were supported by NIH grant R01 LM010098. 
J. B. Cohen was supported by NIH grants K23HL133843 and R01HL153646.  

