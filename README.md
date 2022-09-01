# Pierre-in-Frame
[![Version](https://img.shields.io/badge/version-v0.1-green)](https://github.com/project-pierre/pierre-in-frame) ![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)

[Docs](https://readthedocs.io/) | [Paper](https://doi.org/xx.xxx/xxxxx.xxxxxx)

[![logo](MethodologyWorkflow.svg)](https://project-pierre.github.io/project-pierre)

Pierre-in-Frame is a how-to-create recommendation system focused on multi-objectives.
The framework is a combination of other libraries on recommendation systems and machine learning, such as Scikit-Surprise, Scikit-Pierre and Scikit-learn.
It is constructed to be totally focused on providing a way to produce recommendation systems with more than one objective as calibration, diversity, etc.
In the version 0.1, the framework implements seven collaborative filtering algorithms on the processing and the concept of calibration on the post-processing.
Pierre-in-Frame are composed by **seven steps** in sequence, applying the concept of auto-recsys (auto-ml).
It is totally necessary to run each step following the initial experiment definition because of the concept of auto-recsys.

To create an experiment each step has parameters that receive options from the command line.
Along the steps, the personalization of these parameters can create more than 40 thousand recommendation systems, providing different performance results to the same dataset.
Be cautious with the parameters, if you badly define the configuration a high number of recommender systems will be created, and it is put in a line.

# How-To-Use

## Preparing  
1. Update and upgrade the OS: `sudo apt update && sudo apt upgrade -y && sudo apt install curl`  
2. Download the Anaconda: `curl -O https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh`   
3. Install the Anaconda: `bash Anaconda3-2021.11-Linux-x86_64.sh`  
 
## The code
1. Download the framework: `git clone git@github.com:project-pierre/pierre-in-frame.git`  
2. Go to the framework path: `cd pierre-in-frame/`
3. The framework structure: `ls .`  
3.1. `code/`: Where all python code is. It is the kernel of the framework code.  
3.2. `data/`: Where all data are saved as raw and clean datasets, candidate items, recommendation lists, execution time, etc.  
3.3. `docs/`: The documentation dir to be used on the readthedocs. It is a future work.  
3.4. `environment/`: Where the docker, conda and pip files are. It is used to save all framework environment files.  
3.5. `logs/`: It is produced after the first run. The log path is used to save the code status.  
3.6. `metadata/`: Metadata about the framework.  
3.7. `results/`: The result files will be saved inside this dir, tables, figures, etc.  
3.8. `runners/`: Examples of how-to-run on shell and slurm.  

## Conda Environment
1. Load Conda Environment: `conda env create -f environment/environment.yml`  
2. Start the conda env: `conda activate pierre-in-frame`
3. Or use the script on Ubuntu: `sh environment/installation`  

## Using the framework
To run the framework it is necessary to go to dir: `cd code/`. 
Inside this directory are seven python scripts. Each step has a set of parameters that are necessary to run.    

### Step 1: Pre-processing
The pre-processing step is implemented for cleaning, filtering and modeling the data.
The framework pre-processing the dataset to provide a structure to be used by all next steps.
It applies a data partition in k-fold by n-trials, which k and n are given by the command line.
By default, the framework sets 5-folds and 7-trials.
The pre-processing script provides three parameters to config the experiment:
1. `--dataset=<dataset_name>`: The dataset name declared in the dataset class. By default, the framework provides two dataset pre-installed:  
1.1. Movielens 1M: `ml-1m`  
1.2. Yahoo Movies: `yahoo-movies`  
2. `--n_folds=<number_of_folds>`: The number of folds the user's preferences will be split. By default, the number is 5.  
3. `--n_trials=<number_of_trials>`: The number of trials, one trial is a k-fold, 2 trials are 2 different sets of k-fold. By default, the number is 7.

The train and test file from each fold and trial are created inside:
1. Train: `data/datasets/clean/<dataset_name>/trial-<trial_number>/fold-<fold_number>/train.csv`  
2. Test: `data/datasets/clean/<dataset_name>/trial-<trial_number>/fold-<fold_number>/test.csv`  

#### Run Examples
1. Movielens 1M     
1.1. 10-trials by default 5-folds: `python step1_preprocessing.py --dataset=ml-1m --n_trials=10`  
1.2. 3-folds by default 7-trials: `python step1_preprocessing.py --dataset=ml-1m --n_folds=3`  
1.3. 3-folds by 10-trials: `python step1_preprocessing.py --dataset=ml-1m --n_trials=10 --n_folds=3`  
1.4. Default (5-fold 7-trials): `python step1_preprocessing.py --dataset=ml-1m`  
2. Yahoo Movies  
2.1. 5-trials by default 5-folds: `python step1_preprocessing.py --dataset=yahoo-movies --n_trials=5`  
2.2. 7-folds by default 7-trials: `python step1_preprocessing.py --dataset=yahoo-movies --n_folds=7`  
2.3. 5-folds by 5-trials: `python step1_preprocessing.py --dataset=yahoo-movies --n_trials=5 --n_folds=5`  
2.4. Default (5-fold 7-trials): `python step1_preprocessing.py --dataset=yahoo-movies`  

### Step 2: Best Parameter Search
In this step, the Random Search method is applied to find the best set of hyperparameters.
The best set of parameter values are saved inside the path `code/settings/recommender_param/<dataset_name>/`.
The step provides two parameters to configure the experiment:
1. `--recommender=<recommender_name>`    
The recommender algorithms used in this framework are provided by the Surprise library. All possibilities are collaborative filtering algorithms:    
1.1. `USER_KNN_BASIC`: Normal User KNN.   
1.2. `ITEM_KNN_BASIC`: Normal Item KNN.  
1.3. `SVD`: Singular Value Decomposition.  
1.4. `SVDpp`: Singular Value Decomposition ++.  
1.5. `NMF`: Non-Negative Matrix Factorization.    
1.6. `SLOPE`: Slope One.  
1.7. `CO_CLUSTERING`: Co-Clustering.  
2. `--dataset=<dataset_name>`  
The dataset name declared in the dataset class. By default, the framework provides two dataset pre-installed:  
2.1. Movielens 1M: `ml-1m`  
2.2. Yahoo Movies: `yahoo-movies`    

#### Run Examples
1. Movielens 1M   
1.1. SVD: `python step2_random_search.py --recommender=SVD --dataset=ml-1m`  
1.2. SVDpp: `python step2_random_search.py --recommender=SVDpp --dataset=ml-1m`  
2. Yahoo Movies    
2.1. SVD: `python step2_random_search.py --recommender=SVD --dataset=yahoo-movies`  
2.2. Item: `python step2_random_search.py --recommender=USER_KNN_BASIC --dataset=yahoo-movies`  

### Step 3: Processing
The processing step uses the data pre-processed and the hyperparameters to train the chosen recommender algorithm.
The prediction generates a set of candidate items to be used by the post-processing.
All candidate items set are saved inside `data/app/{dataset}/candidate_items/{recommender}/trial-{trial}/fold-{fold}/candidate_items.csv`.
Its provides an interface with 4 command line parameters that are:  
1. `--recommender=<recommender_name>`: The recommender algorithm name to produce the candidate items set. See the list in the previous step.    
2. `--dataset=<dataset_name>`: The dataset name to be loaded. See the list in the Step 1.  
3. `--fold=<fold_number>`: The fold number to be used as train data. See the list in the Step 1.  
4. `--trial=<trial_number>`: The trial number to be used as train data. See the list in the Step 1.  

If any of these parameters are not given, all the registered or default options will be used in a multiprocessing way.

#### Run Examples
1. Movielens 1M   
1.1. SVD, fold 1 and 1 to 7 trials: `python step3_processing.py --recommender=SVD --dataset=ml-1m --fold=1`  
1.2. SVDpp, all folds(5) and trials(7): `python step3_processing.py --recommender=SVDpp --dataset=ml-1m`  
2. Yahoo Movies    
2.1. SVD, fold 1 and trial 1: `python step3_processing.py --recommender=SVD --dataset=yahoo-movies --fold=1 --trial=1`  
2.2. Item KNN, trial 3 with all folds(5): `python step3_processing.py --recommender=USER_KNN_BASIC --dataset=yahoo-movies --trial=3`  

### Step 4: Post-processing
Post-processing is the focus of this framework. We use the Scikit-Pierre to post-processing the candidate items given by the recommender algorithms provided by the Scikit-Surprise or other recommender library.
The parameters given from the command line are used to create one or more recommendation systems. So, you can change them and create a lot of different systems. It is possible to use the same candidate items set, given as entry to different post-processing formulations.
The recommendations produced by this step are saved in `data/app/<dataset_name>/recommendation_lists/`.
If some parameter is not specified, by default, the framework will apply all options registered to the void parameters.   
This step provides an interface with 11 command line parameters that are:
1. `--recommender=<recommender_name>`: The recommender algorithm name that will be used.
2. `--dataset=<dataset_name>`: The dataset name that will be loaded.
3. `--fold=<fold_number>`: The fold number that will be used to train.
4. `--trial=<trial_number>`: The trial number that will be used to train.

It is necessary to run the algorithm, dataset, fold and trial in step 3 to produce the candidate items set. These 4 parameters are dependent on the previous step, as all the steps. The next parameters will be used to configure the post-processing step.
5. `--tradeoff=<tradeoff_name>`: The tradeoff name that will be constructed based on the next parameters. The Scikit-Pierre provides 2 tradeoffs focused on calibration.
6. `--calibration=<measure_name>`: The measure name that will be used on the tradeoff as a fairness measure. The Scikit-Pierre provides 57 measures·
7. `--relevance=<relevance_name>`: The relevance name that will be used on the tradeoff. The Scikit-Pierre provides 2 measures·
8. `--weight=<weight_name>`: The tradeoff weight that will be used to bring balance between relevance and calibration. The Scikit-Pierre provides 2 personalized ways and accepts constants.
9. `--distribution=<distribution_name>`: The distribution name that will be used to extract the target and realized distributions. The Scikit-Pierre provides 2 distributions·
10. `--selector=<selector_name>`: The selector item algorithm name that will be used to choose the items and creates the recommendation list. The Scikit-Pierre provides the Surrogate Submodular algorithm·
11. `--list_size=<number_of_list_size>`: The number that defines the recommendation list size. The default is 10.

If any of these parameters are not given, all the registered or default options will be used in a multiprocessing way.
The number of CPUs allocated to this job and all jobs on the next steps is N-1.

#### Run Examples
1. Movielens 1M   
1.1. `python step4_postprocessing.py --dataset=ml-1m --trial=1 --fold=1 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`  
1.2. `python step4_postprocessing.py --dataset=ml-1m --trial=3 --fold=5 --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --weight=C@0.5 --calibration=KL`  
1.3. This example `weight` is not given by. The framework will produce 13 recommender systems, variety the weight between constant and personalized: `python step4_postprocessing.py --dataset=ml-1m --trial=3 --fold=5 --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --calibration=KL`  
1.4. This example `calibration` is not given by. The framework will produce 57 recommender systems, variety the measure between similarity and divergence: `python step4_postprocessing.py --dataset=ml-1m --trial=3 --fold=5 --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --weight=C@0.5`  

2. Yahoo Movies    
2.1. `python step4_postprocessing.py --dataset=yahoo-movies --trial=7 --fold=5 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`   
2.2. `python step4_postprocessing.py --dataset=yahoo-movies --trial=2 --fold=1 --recommender=USER_KNN_BASIC --tradeoff=LOG --relevance=SUM --distribution=WPS --weight=CGR --calibration=VICIS_EMANON2`  
2.3. This example `fold` and `trial` are not given by. The framework will produce 35 recommender systems, variety the train and test dataset: `python step4_postprocessing.py --dataset=yahoo-movies --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --calibration=KL`  
2.4. This example `tradeoff` is not given by. The framework will produce 2 recommender systems, variety the equation: `python step4_postprocessing.py --dataset=yahoo-movies --trial=3 --fold=5 --recommender=SVDpp--relevance=SUM --distribution=WPS --weight=C@0.5`  


### Step 5: Metric
In the Metric step, the recommendation lists produced by the post-processing are evaluated.
In version 0.1, the framework provides five metrics: **TIME**, **MAP**, **MRR**, **MACE** and **MRMC**.
All metrics are provided by Scikit-Pierre. The Scikit-Surprise provides other metrics, so it is possible to call these too.
The system that you want to evaluate is given by the same parameters from the post-processing.
The parameter from metric step to choose the metric is `-metric=<metric_initial_in_upper>`.
If the parameter `-metric` is not specified, by default, the framework will apply all registered metrics.
It will happen if some parameter from post-processing is not specified, i.e., if `-metric` are not given by the command line, the step will apply all metrics in all folds from the system specified by the other parameters.  

1. Movielens 1M  
1.1. Mean Rank MisCalibration - MRMC: `python step5_metrics.py -metric=MRMC --dataset=ml-1m --trial=1 --fold=1 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`  
1.2. Mean Average Precision - MAP: `python step5_metrics.py -metric=MAP --dataset=ml-1m --trial=1 --fold=1 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`   
1.3. All metrics: `python step5_metrics.py --dataset=ml-1m --trial=1 --fold=1 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`   

2. Yahoo Movies    
2.1. Mean Reciprocal Rank Precision - MRR: `python step5_metrics.py -metric=MRR --dataset=yahoo-movies --trial=7 --fold=5 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`    
2.2. Mean Average Calibration Error - MACE: `python step5_metrics.py -metric=MACE --dataset=yahoo-movies --trial=7 --fold=5 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`    
2.3. All metrics: `python step5_metrics.py --dataset=yahoo-movies --trial=7 --fold=5 --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`  

### Step 6: Protocol
Protocol step compiles all metrics results, applying an average along the trials and folds.
The final values are the systems performances in each metric.
Coefficients are applied as a new metric. We inherit the coefficients from Scikit-Pierre.
The final file is saved inside: `results/decision/{dataset}/decision.csv`.
The parameters from this step are the same as the Step 4 - post-processing, except the `trial` and `fold`.

#### Run Examples
1. Movielens 1M   
1.1. `python step6_protocol.py --dataset=ml-1m --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`  
1.2. `python step6_protocol.py --dataset=ml-1m --recommender=SVDpp --tradeoff=LOG --relevance=SUM --distribution=WPS --weight=C@0.5 --calibration=KL`    

2. Yahoo Movies    
2.1. `python step6_protocol.py --dataset=yahoo-movies --recommender=SVD --tradeoff=LIN --relevance=NDCG --distribution=CWS --weight=VAR --calibration=COSINE`   
2.2. `python step6_protocol.py --dataset=yahoo-movies --recommender=USER_KNN_BASIC --tradeoff=LOG --relevance=SUM --distribution=WPS --weight=CGR --calibration=VICIS_EMANON2`    

### Step 7: Charts
The Chart step uses the compiled raw results from the Protocol step to produce figures, tables, comparisons and research questions.
It is the last step on the framework. The result files are saved inside: `results/graphics/results/{dataset}/{filename}`.
This step has two parameters `-OPT` and `--dataset`. A set of files are produced based on all metrics from the previous step.
The parameter `-OPT` can be filled as `-OPT=CHART` and `-OPT=ANALYZE`. The option chart generates a set of figures.
The option ‘analyze’ produces prints on the command line that show the best and worst systems combination as well as Welch's t-test.

#### Run Examples
1. Movielens 1M   
1.1. `python step7_charts_tables.py -OPT=CHART --dataset=ml-1m`  
1.2. `python step7_charts_tables.py -OPT=ANALYZE --dataset=ml-1m`    

2. Yahoo Movies    
2.1. `python step7_charts_tables.py -OPT=CHART --dataset=yahoo-movies`   
2.2. `python step7_charts_tables.py -OPT=ANALYZE --dataset=yahoo-movies`    
# unsupervised-frame-of-pierre
